"""
Mock OpenAI-compatible server for testing the Persona harness.

Mimics LM Studio's /v1/models and /v1/chat/completions endpoints.
Generates contextually-aware responses by reading the system prompt
and conversation history, so the harness's adaptation loop can be
tested end-to-end without a real model.

Usage:
    python mock_server.py          # starts on port 1234
    python mock_server.py --port 5000
"""

import json
import re
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

MODEL_ID = "mock-qwen-for-testing"

# ---------------------------------------------------------------------------
# Canned response logic
# ---------------------------------------------------------------------------

# Simple pattern-matched responses that a "friendly chat partner" might give.
# These are intentionally varied so the harness's profiling has real signal.
TOPIC_RESPONSES = {
    "gaming": [
        "Oh nice, what games you been playing lately? I've been getting into some roguelikes recently.",
        "Gaming is such a good way to unwind honestly. What's your go-to genre?",
    ],
    "coding": [
        "What language are you working in? I've been messing around with Rust lately and it's been a ride.",
        "Yeah coding can be super satisfying when things click. What are you building?",
    ],
    "school": [
        "Ugh school can be rough. What classes are you taking this semester?",
        "How's the workload been? Finals season is always brutal.",
    ],
    "studying": [
        "What subject? Sometimes just switching up your study spot helps a ton.",
        "Flashcards or practice problems? Depends on the subject I guess.",
    ],
    "dating": [
        "Dating scene is wild these days lol. You on the apps or meeting people irl?",
        "What happened? Sometimes you gotta just put yourself out there.",
    ],
    "fitness": [
        "Nice, what's your routine looking like? I've been trying to be more consistent.",
        "Getting into fitness is one of the best things you can do honestly. What got you started?",
    ],
    "grades": [
        "What class? Sometimes talking to the professor directly helps more than you'd think.",
        "Don't stress too hard about it. One bad grade doesn't define anything.",
    ],
    "keyboards": [
        "Oh you're into keebs? What switches do you run?",
        "Mechanical keyboards are such a rabbit hole lol. What's your daily driver?",
    ],
}

GENERIC_RESPONSES = [
    "That's interesting, tell me more about that.",
    "Yeah I feel that. What's been on your mind?",
    "Hmm I hadn't thought about it that way. What made you think of that?",
    "For real though, that makes a lot of sense.",
    "I get what you mean. How long has that been going on?",
    "Honestly that's a solid take. What do you think you'll do about it?",
    "Oh word? That's cool. How'd you get into that?",
    "I hear you. Sometimes things just be like that.",
]

SLANG_RESPONSES = [
    "lmao fr fr, that's wild",
    "bruh no way, tell me more",
    "nah cuh that's crazy, what happened next?",
    "lol idk man that's tough",
    "haha yeah blud I feel you on that",
]

PROFILER_TEMPLATE = """COGNITIVE: {cog}
ENGAGEMENT: {eng}
INTERESTS: {interests}
SLANG: {slang}"""


def _pick_response(responses, seed):
    """Deterministically pick from a list using a seed."""
    return responses[seed % len(responses)]


def _detect_topics(text):
    """Find topic keywords in user text."""
    text_lower = text.lower()
    found = []
    for topic in TOPIC_RESPONSES:
        if topic in text_lower:
            found.append(topic)
    return found


def _is_profiler_request(messages):
    """Check if this is a profiler analysis request (not a regular chat)."""
    for msg in messages:
        if msg.get("role") == "user" and "analyze this transcript" in msg.get("content", "").lower():
            return True
    return False


def _extract_transcript_for_profiling(messages):
    """Pull the transcript text out of a profiler request."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _generate_profiler_response(transcript_text):
    """Generate a mock profiler analysis based on the transcript content."""
    text_lower = transcript_text.lower()

    # Cognitive axis: look for emotional vs logical language
    emotional_words = ["feel", "love", "hate", "happy", "sad", "stress", "mood", "vibe", "ugh"]
    logical_words = ["think", "reason", "data", "spec", "performance", "benchmark", "compare"]
    e_score = sum(1 for w in emotional_words if w in text_lower)
    s_score = sum(1 for w in logical_words if w in text_lower)
    cog = "E" if e_score >= s_score else "S"

    # Engagement axis: look for question marks and conversational openers
    question_marks = text_lower.count("?")
    eng = "P" if question_marks >= 2 else "D"

    # Interests: detect topics
    # Map variations to canonical interest keywords
    interest_aliases = {
        "gaming": ["gaming", "games", "game", "fps", "rpg", "mmorpg", "fortnite", "valorant"],
        "coding": ["coding", "code", "python", "javascript", "rust", "algorithms"],
        "programming": ["programming", "programmer", "developer", "software"],
        "school": ["school", "class", "classes", "semester", "professor"],
        "studying": ["studying", "study", "spaced repetition", "flashcard"],
        "dating": ["dating", "date", "tinder", "bumble", "hinge"],
        "fitness": ["fitness", "gym", "workout", "exercise", "lifting"],
        "grades": ["grades", "grade", "gpa", "declining"],
        "keyboards": ["keyboards", "keyboard", "keebs", "switches", "mechanical"],
        "running": ["running", "run", "marathon", "jogging"],
        "exams": ["exams", "exam", "final", "finals", "midterm", "test"],
        "relationships": ["relationships", "relationship", "girlfriend", "boyfriend"],
    }
    found_interests = []
    for canonical, aliases in interest_aliases.items():
        if any(alias in text_lower for alias in aliases):
            found_interests.append(canonical)
    interests_str = ", ".join(found_interests[:3]) if found_interests else "none"

    # Slang detection
    slang_patterns = ["lol", "btw", "haha", "gonna", "cuh", "lmao", "idk", "blud", "fr fr", "bruh"]
    found_slang = [s for s in slang_patterns if s in text_lower]
    slang_str = ", ".join(found_slang[:4]) if found_slang else "none"

    return PROFILER_TEMPLATE.format(cog=cog, eng=eng, interests=interests_str, slang=slang_str)


def _generate_chat_response(system_prompt, messages):
    """Generate a contextually-aware mock chat response."""
    # Get the last user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return "Hey, what's up?"

    last_msg = user_messages[-1]["content"]
    seed = len(last_msg) + sum(ord(c) for c in last_msg[:20])

    # Check system prompt for style cues
    sys_lower = system_prompt.lower() if system_prompt else ""
    is_casual = "casual" in sys_lower or "informal" in sys_lower
    is_concise = "short" in sys_lower or "concise" in sys_lower or "1 to 2 sentences" in sys_lower
    has_slang_directive = "slang" in sys_lower

    # Detect topics in user message
    topics = _detect_topics(last_msg)

    # Pick response
    if topics:
        topic = topics[0]
        response = _pick_response(TOPIC_RESPONSES[topic], seed)
    elif has_slang_directive and is_casual:
        response = _pick_response(SLANG_RESPONSES, seed)
    else:
        response = _pick_response(GENERIC_RESPONSES, seed)

    # Trim if concise mode
    if is_concise and len(response) > 80:
        # Just take first sentence
        first_sentence_end = response.find(".")
        if first_sentence_end > 0:
            response = response[: first_sentence_end + 1]

    return response


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class MockLMStudioHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible API requests."""

    def log_message(self, format, *args):
        """Custom log format with timestamp."""
        sys.stderr.write(f"[MockServer] {args[0]}\n")

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "mock"}],
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))

            messages = body.get("messages", [])
            system_prompt = ""
            chat_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                else:
                    chat_messages.append(msg)

            # Route to profiler or chat response generator
            if _is_profiler_request(chat_messages):
                transcript = _extract_transcript_for_profiling(chat_messages)
                response_text = _generate_profiler_response(transcript)
            else:
                response_text = _generate_chat_response(system_prompt, chat_messages)

            # Simulate a small delay for realism
            time.sleep(0.3)

            self._send_json({
                "id": f"mock-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        else:
            self._send_json({"error": "Not found"}, 404)


def main():
    port = 1234
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    server = HTTPServer(("127.0.0.1", port), MockLMStudioHandler)
    print(f"[MockServer] Mock LM Studio server running on http://127.0.0.1:{port}")
    print(f"[MockServer] Model ID: {MODEL_ID}")
    print(f"[MockServer] Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MockServer] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
