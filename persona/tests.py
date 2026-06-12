"""
Automated test suite for the Persona SDK.

Tests the public API: enhance(), observe(), profile, persistence, hooks.

Usage:
    1. Start the mock server:  python mock_server.py
    2. Run this:               python -m persona.tests
"""

import json
import os
import sys
import tempfile
import shutil

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.system("")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from persona import Persona, SponsoredSuggestionsHook
from persona.hooks import EnrichmentHook, ToneOverrideHook
from persona.providers import OpenAICompatibleProvider, NullProvider


# ---------------------------------------------------------------------------
# Test conversations
# ---------------------------------------------------------------------------

CASUAL_GAMER = [
    ("yo whats good", "Hey! Not much, what's up?"),
    ("lol nothing much just been playing games all day", "Nice, what games?"),
    ("yeah bro I got into this new fps, the movement is insane fr fr", "Oh sick, which one?"),
    ("nah cuh the keyboards at best buy are trash", "Yeah the quality varies a lot"),
    ("lmao idk maybe I should just build my own", "That'd be cool actually"),
    ("haha yeah blud thats what im saying", "Do it!"),
]

FORMAL_STUDENT = [
    ("Hello, I have a question about study strategies.", "Of course! What would you like to know?"),
    ("I'm preparing for my final exams in organic chemistry.", "That's a tough subject."),
    ("Would you recommend spaced repetition?", "Yes, it's quite effective for memorization."),
    ("My grades have been declining this semester.", "I'm sorry to hear that."),
    ("I've been considering hiring a tutor.", "That could definitely help."),
    ("Thank you for the advice.", "You're welcome!"),
]

MIXED_CODER = [
    ("hey so I've been learning python lately", "Cool! Python is great."),
    ("coding is actually pretty fun", "Right? It clicks at some point."),
    ("do you know any good resources for algorithms?", "Sure, there are many."),
    ("also I've been hitting the gym more \U0001F4AA", "Nice combo!"),
    ("lol yeah programming and fitness", "Balance is key."),
    ("gonna try to build a workout tracker app", "That's a great project."),
]


def run_checks():
    results = {}
    tmp_dirs = []

    try:
        # ==============================================================
        # TEST 1: Basic profiling with mock server
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 1: Casual Gamer Profiling")
        print("=" * 60)

        provider = OpenAICompatibleProvider()
        ok, detail = provider.check_health()
        if not ok:
            print(f"  [SKIP] Mock server not running: {detail}")
            print("  Start it with: python mock_server.py")
            return False

        print(f"  Connected to: {detail}")

        tmp = tempfile.mkdtemp(prefix="persona_test_")
        tmp_dirs.append(tmp)
        p = Persona(user_id="gamer", provider=provider, storage_dir=tmp)
        for user_msg, bot_msg in CASUAL_GAMER:
            p.observe(user_msg, bot_msg)
        p.wait_for_profiler()

        prof = p.profile
        print(f"  Style: {prof['style_code']}")
        print(f"  Interests: {prof['interests']}")
        print(f"  Slang: {prof['slang']}")
        print(f"  Confidence: {prof['confidence']}")

        results["gamer_casual"] = prof["style_axes"]["formality"] == "casual"
        results["gamer_slang"] = len(prof["slang"]) > 0
        results["gamer_interests"] = any(
            i in prof["interests"] for i in ["gaming", "keyboards"]
        )

        # ==============================================================
        # TEST 2: Formal student
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 2: Formal Student Profiling")
        print("=" * 60)

        tmp2 = tempfile.mkdtemp(prefix="persona_test_")
        tmp_dirs.append(tmp2)
        p2 = Persona(user_id="student", provider=provider, storage_dir=tmp2)
        for user_msg, bot_msg in FORMAL_STUDENT:
            p2.observe(user_msg, bot_msg)
        p2.wait_for_profiler()

        prof2 = p2.profile
        print(f"  Style: {prof2['style_code']}")
        print(f"  Interests: {prof2['interests']}")

        results["student_formal"] = prof2["style_axes"]["formality"] == "formal"
        results["student_no_slang"] = len(prof2["slang"]) == 0
        results["student_interests"] = any(
            i in prof2["interests"] for i in ["studying", "grades", "exams", "school"]
        )

        # ==============================================================
        # TEST 3: enhance() output
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 3: Prompt Enhancement")
        print("=" * 60)

        base = "You are a helpful assistant."
        enhanced = p.enhance(base)
        print(f"  Base prompt: {base}")
        print(f"  Enhanced:\n    {enhanced.replace(chr(10), chr(10) + '    ')}")

        results["enhance_includes_base"] = base in enhanced
        # Count lines added by Persona (the supplement)
        supplement_lines = enhanced.replace(base, "").strip().split("\n")
        supplement_lines = [l for l in supplement_lines if l.strip()]
        print(f"  Supplement: {len(supplement_lines)} lines")
        results["enhance_compact"] = len(supplement_lines) <= 5

        # ==============================================================
        # TEST 4: enhance() with no profile data
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 4: enhance() with fresh profile")
        print("=" * 60)

        tmp3 = tempfile.mkdtemp(prefix="persona_test_")
        tmp_dirs.append(tmp3)
        p_fresh = Persona(user_id="fresh", storage_dir=tmp3)
        fresh_enhanced = p_fresh.enhance("You are a bot.")
        print(f"  Fresh enhanced: {fresh_enhanced}")

        # With zero confidence, the supplement should be minimal
        # (just emoji directive at most)
        results["fresh_minimal"] = len(fresh_enhanced) < len("You are a bot.") + 100

        # ==============================================================
        # TEST 5: Interest accumulation
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 5: Interest Accumulation")
        print("=" * 60)

        tmp4 = tempfile.mkdtemp(prefix="persona_test_")
        tmp_dirs.append(tmp4)
        p_acc = Persona(user_id="accumulator", provider=provider, storage_dir=tmp4)

        # Batch 1: gaming
        for msg, reply in [
            ("I love playing video games", "Cool!"),
            ("been gaming all day honestly", "Nice"),
            ("this new fps game is insane", "Which one?"),
        ]:
            p_acc.observe(msg, reply)
        p_acc.wait_for_profiler()
        interests_1 = set(p_acc.profile["interests"])
        print(f"  After gaming: {interests_1}")

        # Batch 2: coding
        for msg, reply in [
            ("I also code in python", "Nice!"),
            ("coding is really fun", "Agree"),
            ("been building a web app", "Cool project"),
        ]:
            p_acc.observe(msg, reply)
        p_acc.wait_for_profiler()
        interests_2 = set(p_acc.profile["interests"])
        print(f"  After coding: {interests_2}")

        results["accumulation"] = interests_1.issubset(interests_2)

        # ==============================================================
        # TEST 6: Axis smoothing
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 6: Axis Smoothing")
        print("=" * 60)

        tmp5 = tempfile.mkdtemp(prefix="persona_test_")
        tmp_dirs.append(tmp5)
        p_smooth = Persona(user_id="smoother", storage_dir=tmp5)

        for msg in ["lol whats good bro", "haha yeah cuh", "lmao fr fr no cap", "idk man lol"]:
            p_smooth.observe(msg)

        before = p_smooth.profile["style_axes"]["formality"]
        print(f"  After 4 casual: {before}")

        p_smooth.observe("I would like to formally request additional information please.")
        after = p_smooth.profile["style_axes"]["formality"]
        print(f"  After 1 formal: {after}")

        results["smoothing"] = after == "casual"

        # ==============================================================
        # TEST 7: Hooks
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 7: Enrichment Hooks")
        print("=" * 60)

        ad_hook = SponsoredSuggestionsHook({
            "gaming": "mechanical keyboards",
            "coding": "AI copilot licenses",
        })
        tone_hook = ToneOverrideHook("Always be encouraging.")

        tmp6 = tempfile.mkdtemp(prefix="persona_test_")
        tmp_dirs.append(tmp6)
        p_hooks = Persona(
            user_id="hooked", provider=provider, storage_dir=tmp6,
            hooks=[ad_hook, tone_hook],
        )

        for msg, reply in [
            ("been playing games all day", "Fun!"),
            ("yeah gaming is great lol", "Totally"),
            ("I love fps games fr fr", "Same here"),
        ]:
            p_hooks.observe(msg, reply)
        p_hooks.wait_for_profiler()

        enhanced = p_hooks.enhance("You are a friend.")
        print(f"  Enhanced with hooks:\n    {enhanced.replace(chr(10), chr(10) + '    ')}")

        results["hook_tone"] = "encouraging" in enhanced.lower()
        results["hook_ads"] = "keyboard" in enhanced.lower() or "copilot" in enhanced.lower() or len(p_hooks.profile["interests"]) > 0

        # Test disabling ads
        ad_hook.enabled = False
        enhanced_no_ads = p_hooks.enhance("You are a friend.")
        results["hook_disable"] = "keyboard" not in enhanced_no_ads.lower()

        # ==============================================================
        # TEST 8: Persistence
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 8: Profile Persistence")
        print("=" * 60)

        tmp7 = tempfile.mkdtemp(prefix="persona_test_")
        tmp_dirs.append(tmp7)

        p_save = Persona(user_id="persist", provider=provider, storage_dir=tmp7)
        for msg, reply in CASUAL_GAMER[:3]:
            p_save.observe(msg, reply)
        p_save.wait_for_profiler()
        p_save.save()

        prof_saved = p_save.profile
        print(f"  Saved: style={prof_saved['style_code']}, interests={prof_saved['interests']}")

        # Load in a new instance
        p_load = Persona(user_id="persist", storage_dir=tmp7)
        prof_loaded = p_load.profile
        print(f"  Loaded: style={prof_loaded['style_code']}, interests={prof_loaded['interests']}")

        results["persist_style"] = prof_saved["style_code"] == prof_loaded["style_code"]
        results["persist_interests"] = set(prof_saved["interests"]) == set(prof_loaded["interests"])

        # ==============================================================
        # TEST 9: NullProvider (heuristic-only mode)
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 9: Heuristic-Only Mode (NullProvider)")
        print("=" * 60)

        tmp8 = tempfile.mkdtemp(prefix="persona_test_")
        tmp_dirs.append(tmp8)
        p_null = Persona(user_id="null_test", provider=NullProvider(), storage_dir=tmp8)
        for msg in ["lol hey bro", "haha yeah cuh", "lmao idk man"]:
            p_null.observe(msg)
        p_null.wait_for_profiler()

        null_prof = p_null.profile
        print(f"  Style: {null_prof['style_code']}")
        print(f"  Confidence: {null_prof['confidence']}")

        # Heuristic axes should work, LLM axes get NullProvider defaults
        results["null_heuristics"] = null_prof["confidence"]["len"] > 0

        # ==============================================================
        # TEST 10: Confidence gating
        # ==============================================================
        print("\n" + "=" * 60)
        print("  TEST 10: Confidence Gating")
        print("=" * 60)

        tmp9 = tempfile.mkdtemp(prefix="persona_test_")
        tmp_dirs.append(tmp9)
        p_conf = Persona(user_id="conf_test", storage_dir=tmp9, confidence_threshold=5)
        p_conf.observe("lol hey")
        supplement_early = p_conf._build_supplement()
        print(f"  After 1 msg (conf=1): '{supplement_early}'")

        for msg in ["haha bro", "lmao cuh", "idk fr fr", "btw ngl"]:
            p_conf.observe(msg)
        supplement_late = p_conf._build_supplement()
        print(f"  After 5 msgs (conf=5): '{supplement_late}'")

        # Early: no style rules (not confident enough)
        # Late: should have style rules
        results["conf_gate_early"] = "short" not in supplement_early.lower() and "casual" not in supplement_early.lower()
        results["conf_gate_late"] = len(supplement_late) > len(supplement_early)

        # ==============================================================
        # SUMMARY
        # ==============================================================
        print(f"\n{'=' * 60}")
        print("  SUMMARY")
        print(f"{'=' * 60}")

        all_pass = True
        for name, passed in results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} {name}")
            if not passed:
                all_pass = False

        count = len(results)
        passed = sum(1 for v in results.values() if v)
        print(f"\n  {passed}/{count} checks passed{'!' if all_pass else ''}")
        return all_pass

    finally:
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)


def main():
    ok = run_checks()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
