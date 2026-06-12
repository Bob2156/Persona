"""
Persona engine — the core SDK class.

This is the only class most customers need to interact with.

Usage:
    from persona import Persona

    p = Persona(user_id="user_123")

    # Enhance any system prompt with personalization
    enhanced = p.enhance("You are a helpful assistant.")

    # After each turn, feed observations back
    p.observe(user_msg="lol whats good", assistant_msg="Hey! Not much.")

    # Profile is available anytime
    print(p.profile)
"""

import json
import os
import threading
from collections import deque

from persona.defaults import (
    CONFIDENCE_THRESHOLD,
    MAX_INTERESTS,
    MAX_SLANG,
    PROFILER_INTERVAL,
    PROFILER_TRANSCRIPT_WINDOW,
    SMOOTHING_WINDOW,
    STYLE_RULES,
)
from persona.profiling import (
    build_profiler_system_prompt,
    compute_heuristics,
    format_transcript,
    parse_profiler_response,
)


class Persona:
    """Real-time user personality profiling middleware.

    Persona passively builds a behavioral profile as users chat and
    generates a compact personalization supplement for any system prompt.

    The profile includes:
    - Communication style (4 axes: length, cognitive, formality, engagement)
    - Accumulated interests
    - Detected slang vocabulary
    - Emoji usage preference
    - Confidence scores per axis

    The profiler can optionally make background LLM calls for deeper
    analysis. If no provider is given, only fast heuristics are used.

    Args:
        user_id: Unique identifier for this user's profile.
        provider: A ProfilerProvider for background LLM analysis.
                  Pass None or NullProvider for heuristic-only mode.
        storage_dir: Directory where profiles are persisted. Set to None
                     to disable persistence.
        hooks: List of EnrichmentHook instances for prompt enrichment.
        profiler_interval: Run deep profiler every N user messages.
        smoothing_window: Number of recent observations per axis.
        confidence_threshold: Min observations before injecting style rules.
    """

    def __init__(
        self,
        user_id="default",
        provider=None,
        storage_dir=None,
        hooks=None,
        profiler_interval=PROFILER_INTERVAL,
        smoothing_window=SMOOTHING_WINDOW,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    ):
        self.user_id = user_id
        self.provider = provider
        self.hooks = list(hooks) if hooks else []

        # Config
        self._profiler_interval = profiler_interval
        self._smoothing_window = smoothing_window
        self._confidence_threshold = confidence_threshold

        # Storage
        self._storage_dir = storage_dir
        self._profile_path = None
        if storage_dir is not None:
            os.makedirs(storage_dir, exist_ok=True)
            self._profile_path = os.path.join(storage_dir, f"{user_id}.json")

        # --- Axis smoothing ---
        self._axis_history = {
            "len": deque(maxlen=smoothing_window),
            "cog": deque(maxlen=smoothing_window),
            "form": deque(maxlen=smoothing_window),
            "eng": deque(maxlen=smoothing_window),
        }
        self._mbti_letters = {"len": "C", "cog": "E", "form": "I", "eng": "D"}

        # --- Accumulated profile data ---
        self._interests = set()
        self._slang = set()
        self._uses_emojis = False

        # --- Conversation state ---
        self._conversation_history = []
        self._user_message_counter = 0

        # --- Thread safety ---
        self._lock = threading.Lock()
        self._profiler_thread = None

        # Load saved profile
        self._load()

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def enhance(self, base_system_prompt=""):
        """Enhance a system prompt with personalization.

        This is the primary integration point. Call this before each
        LLM request to get a personalized system prompt.

        Args:
            base_system_prompt: The customer's original system prompt.

        Returns:
            The original prompt with a compact personalization supplement
            appended. If no profile data is available yet, returns the
            original prompt unchanged.
        """
        supplement = self._build_supplement()
        if not supplement:
            return base_system_prompt

        if base_system_prompt:
            return f"{base_system_prompt}\n{supplement}"
        return supplement

    def observe(self, user_msg, assistant_msg=None):
        """Observe a conversation turn to update the profile.

        Call this after each user message (and optionally the assistant
        response) to feed data into the profiling engine.

        Args:
            user_msg: The user's message text.
            assistant_msg: The assistant's response text (optional but
                          improves profiler accuracy).
        """
        with self._lock:
            self._conversation_history.append({"role": "user", "content": user_msg})
            if assistant_msg:
                self._conversation_history.append({"role": "assistant", "content": assistant_msg})

        # Run fast heuristics on every message
        self._update_heuristics()

        # Trigger deep profiler periodically
        with self._lock:
            self._user_message_counter += 1
            should_profile = self._user_message_counter >= self._profiler_interval
            if should_profile:
                self._user_message_counter = 0

        if should_profile and self.provider is not None:
            self._trigger_profiler()

    @property
    def profile(self):
        """Return a snapshot of the current user profile."""
        with self._lock:
            return {
                "user_id": self.user_id,
                "style_code": self._current_style_code,
                "style_axes": {
                    "length": "concise" if self._mbti_letters["len"] == "C" else "verbose",
                    "cognitive": "emotion-aware" if self._mbti_letters["cog"] == "E" else "logic-first",
                    "formality": "casual" if self._mbti_letters["form"] == "I" else "formal",
                    "engagement": "direct" if self._mbti_letters["eng"] == "D" else "open-ended",
                },
                "confidence": {
                    axis: len(self._axis_history[axis])
                    for axis in ("len", "cog", "form", "eng")
                },
                "interests": sorted(self._interests),
                "slang": sorted(self._slang),
                "uses_emojis": self._uses_emojis,
                "messages_observed": len(self._conversation_history),
            }

    def reset(self):
        """Reset all profile state and conversation history."""
        with self._lock:
            self._axis_history = {
                axis: deque(maxlen=self._smoothing_window)
                for axis in ("len", "cog", "form", "eng")
            }
            self._mbti_letters = {"len": "C", "cog": "E", "form": "I", "eng": "D"}
            self._interests = set()
            self._slang = set()
            self._uses_emojis = False
            self._conversation_history = []
            self._user_message_counter = 0
        if self._profiler_thread and self._profiler_thread.is_alive():
            self._profiler_thread.join(timeout=0.5)

    def save(self):
        """Manually save the profile to disk."""
        self._save()

    def wait_for_profiler(self, timeout=10):
        """Block until the background profiler finishes (for testing)."""
        if self._profiler_thread and self._profiler_thread.is_alive():
            self._profiler_thread.join(timeout=timeout)

    # ==================================================================
    # INTERNAL — Axis smoothing
    # ==================================================================

    @property
    def _current_style_code(self):
        return (
            f"{self._mbti_letters['len']}"
            f"{self._mbti_letters['cog']}"
            f"{self._mbti_letters['form']}"
            f"{self._mbti_letters['eng']}"
        )

    def _observe_axis(self, axis, letter):
        """Record an observation and resolve via majority vote."""
        self._axis_history[axis].append(letter)
        history = self._axis_history[axis]
        if not history:
            return
        counts = {}
        for l in history:
            counts[l] = counts.get(l, 0) + 1
        best = max(counts, key=counts.get)
        current = self._mbti_letters[axis]
        # Stability bias: only flip if strictly more votes
        if counts[best] > counts.get(current, 0):
            self._mbti_letters[axis] = best

    # ==================================================================
    # INTERNAL — Heuristics
    # ==================================================================

    def _update_heuristics(self):
        with self._lock:
            user_msgs = [m for m in self._conversation_history if m["role"] == "user"]
            recent = user_msgs[-3:]

        result = compute_heuristics(recent)
        if not result:
            return

        len_letter, form_letter, uses_emojis = result
        with self._lock:
            self._observe_axis("len", len_letter)
            self._observe_axis("form", form_letter)
            self._uses_emojis = uses_emojis

    # ==================================================================
    # INTERNAL — Deep profiler
    # ==================================================================

    def _trigger_profiler(self):
        if self._profiler_thread and self._profiler_thread.is_alive():
            return
        self._profiler_thread = threading.Thread(
            target=self._run_profiler, daemon=True
        )
        self._profiler_thread.start()

    def _run_profiler(self):
        with self._lock:
            transcript = format_transcript(
                self._conversation_history,
                recent_count=PROFILER_TRANSCRIPT_WINDOW,
            )

        system_prompt = build_profiler_system_prompt()
        response = self.provider.complete(
            system_prompt, f"Analyze this transcript:\n\n{transcript}"
        )

        results = parse_profiler_response(response)
        with self._lock:
            if results["cog"]:
                self._observe_axis("cog", results["cog"])
            if results["eng"]:
                self._observe_axis("eng", results["eng"])
            if results["interests"]:
                self._interests.update(results["interests"])
                if len(self._interests) > MAX_INTERESTS:
                    self._interests = set(sorted(self._interests)[-MAX_INTERESTS:])
            if results["slang"]:
                self._slang.update(results["slang"])
                if len(self._slang) > MAX_SLANG:
                    self._slang = set(sorted(self._slang)[-MAX_SLANG:])

        self._save()

    # ==================================================================
    # INTERNAL — Prompt supplement
    # ==================================================================

    def _build_supplement(self):
        """Build the compact personalization supplement."""
        with self._lock:
            letters = dict(self._mbti_letters)
            confidence = {
                axis: len(self._axis_history[axis])
                for axis in ("len", "cog", "form", "eng")
            }
            slang = sorted(self._slang)
            uses_emojis = self._uses_emojis
            # Build a snapshot inline (can't call self.profile — same lock)
            profile_snapshot = {
                "user_id": self.user_id,
                "interests": sorted(self._interests),
                "slang": slang,
                "uses_emojis": uses_emojis,
            }

        # Collect style rules for confident axes only
        style_parts = []
        for axis in ("len", "cog", "form", "eng"):
            if confidence[axis] >= self._confidence_threshold:
                rule = STYLE_RULES.get(letters[axis])
                if rule:
                    style_parts.append(rule)

        lines = []

        # Style line
        if style_parts:
            lines.append(" ".join(style_parts))

        # Vocabulary line
        vocab_parts = []
        if slang:
            slang_str = ", ".join(f"'{s}'" for s in slang[:5])
            vocab_parts.append(f"Mirror their slang: {slang_str}.")
        if uses_emojis:
            vocab_parts.append("Use emojis naturally.")
        else:
            vocab_parts.append("No emojis.")
        if vocab_parts:
            lines.append(" ".join(vocab_parts))

        # Hook enrichments
        for hook in self.hooks:
            try:
                enrichment = hook.enrich(profile_snapshot)
                if enrichment:
                    lines.append(enrichment)
            except Exception:
                pass  # Hooks should never crash the engine

        return "\n".join(lines) if lines else ""

    # ==================================================================
    # INTERNAL — Persistence
    # ==================================================================

    def _save(self):
        if not self._profile_path:
            return
        with self._lock:
            data = {
                "user_id": self.user_id,
                "mbti_letters": dict(self._mbti_letters),
                "axis_history": {
                    axis: list(h) for axis, h in self._axis_history.items()
                },
                "interests": sorted(self._interests),
                "slang": sorted(self._slang),
                "uses_emojis": self._uses_emojis,
            }
        try:
            with open(self._profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _load(self):
        if not self._profile_path or not os.path.exists(self._profile_path):
            return
        try:
            with open(self._profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                if "mbti_letters" in data:
                    self._mbti_letters.update(data["mbti_letters"])
                if "axis_history" in data:
                    for axis, history in data["axis_history"].items():
                        if axis in self._axis_history:
                            self._axis_history[axis] = deque(
                                history, maxlen=self._smoothing_window
                            )
                if "interests" in data:
                    self._interests = set(data["interests"])
                if "slang" in data:
                    self._slang = set(data["slang"])
                if "uses_emojis" in data:
                    self._uses_emojis = data["uses_emojis"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
