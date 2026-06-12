"""
Automated test driver for the Persona harness.

Simulates multi-turn conversations against the mock server (or real LM Studio),
exercises the profiling loop, and validates:
  - Interest accumulation (merge, not replace)
  - Axis smoothing (majority-vote stability)
  - Confidence scores
  - Compressed prompt length
  - Profile persistence (save/load)

Usage:
    1. Start the mock server:  python mock_server.py
    2. Run this:               python test_harness.py
"""

import json
import os
import sys
import tempfile

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.system("")  # enable VT100
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from harness import DynamicChatHarness
from lm_studio_client import LMStudioClient, check_server_status


# ---------------------------------------------------------------------------
# Test conversations — each is a sequence of user messages designed to
# trigger specific profiling behaviors.
# ---------------------------------------------------------------------------

CASUAL_GAMER_CONVERSATION = [
    "yo whats good",
    "lol nothing much just been playing games all day",
    "yeah bro I got into this new fps, the movement is insane fr fr",
    "nah cuh the keyboards at best buy are trash, I need a good mechanical one",
    "lmao idk maybe I should just build my own",
    "haha yeah blud thats what im saying",
]

FORMAL_STUDENT_CONVERSATION = [
    "Hello, I have a question about effective study strategies.",
    "I'm currently preparing for my final exams in organic chemistry.",
    "Would you recommend spaced repetition for memorizing reaction mechanisms?",
    "My grades have been declining this semester and I'm quite concerned.",
    "I've been considering hiring a tutor. What do you think?",
    "Thank you for the advice. I'll implement those changes immediately.",
]

MIXED_CODER_CONVERSATION = [
    "hey so I've been learning python lately",
    "coding is actually pretty fun once you get the hang of it",
    "do you know any good resources for learning algorithms?",
    "also I've been hitting the gym more, trying to get in shape \U0001F4AA",
    "lol yeah programming and fitness, weird combo right?",
    "gonna try to build a workout tracker app actually",
]


def print_separator(char="\u2500", width=60):
    print(char * width)


def print_profile_diff(before, after):
    """Show what changed between two profile snapshots."""
    changes = []
    if before["style_code"] != after["style_code"]:
        changes.append(f"  Style: {before['style_code']} \u2192 {after['style_code']}")
    if before["interests"] != after["interests"]:
        changes.append(f"  Interests: {before['interests']} \u2192 {after['interests']}")
    if before["slang"] != after["slang"]:
        changes.append(f"  Slang: {before['slang']} \u2192 {after['slang']}")
    if before["uses_emojis"] != after["uses_emojis"]:
        changes.append(f"  Emojis: {before['uses_emojis']} \u2192 {after['uses_emojis']}")
    if before.get("potential_sponsored_suggestions") != after.get("potential_sponsored_suggestions"):
        changes.append(f"  Ads: {before.get('potential_sponsored_suggestions', [])} \u2192 {after.get('potential_sponsored_suggestions', [])}")
    if changes:
        print("  \U0001F4CA Profile changes:")
        for c in changes:
            print(c)
    else:
        print("  \U0001F4CA No profile changes")


def run_conversation(name, messages, client, profile_path=None):
    """Run a simulated conversation and report profiling results."""
    print(f"\n{'=' * 60}")
    print(f"  TEST: {name}")
    print(f"{'=' * 60}")

    kwargs = {}
    if profile_path:
        kwargs["profile_path"] = profile_path
    harness = DynamicChatHarness(client, **kwargs)

    for i, msg in enumerate(messages, 1):
        before = harness.profile_snapshot()

        print(f"\n  [{i}/{len(messages)}] User: {msg}")
        reply = harness.send_chat_message(msg)
        print(f"  Bot: {reply}")

        # Wait for any background profiler thread to finish
        if harness._profiler_thread and harness._profiler_thread.is_alive():
            print("  \u23F3 Waiting for background profiler...")
            harness._profiler_thread.join(timeout=10)

        after = harness.profile_snapshot()
        print_profile_diff(before, after)

    # Final profile
    print_separator("\u2500")
    final = harness.profile_snapshot()
    print("  \U0001F3C1 Final Profile:")
    print(f"     Style Code:  {final['style_code']}")
    print(f"     Axes:        {json.dumps(final['style_axes'])}")
    print(f"     Confidence:  {json.dumps(final.get('confidence', {}))}")
    print(f"     Interests:   {final['interests']}")
    print(f"     Slang:       {final['slang']}")
    print(f"     Emojis:      {final['uses_emojis']}")
    print(f"     Ads:         {final['potential_sponsored_suggestions']}")
    print(f"     Messages:    {final['messages_in_context']}")

    return final, harness


def test_interest_accumulation(client):
    """Verify that interests accumulate across profiler runs instead of replacing."""
    print(f"\n{'=' * 60}")
    print("  TEST: Interest Accumulation")
    print(f"{'=' * 60}")

    harness = DynamicChatHarness(client, profile_path=os.devnull)

    # First batch: talk about gaming
    gaming_msgs = [
        "I love playing video games",
        "been gaming all day honestly",
        "this new fps game is insane",
    ]
    for msg in gaming_msgs:
        harness.send_chat_message(msg)
    if harness._profiler_thread and harness._profiler_thread.is_alive():
        harness._profiler_thread.join(timeout=10)
    interests_after_gaming = set(harness.profile_snapshot()["interests"])
    print(f"  After gaming msgs: {interests_after_gaming}")

    # Second batch: talk about coding (should ADD, not replace)
    coding_msgs = [
        "I also code in python",
        "coding is really fun",
        "been building a web app",
    ]
    for msg in coding_msgs:
        harness.send_chat_message(msg)
    if harness._profiler_thread and harness._profiler_thread.is_alive():
        harness._profiler_thread.join(timeout=10)
    interests_after_coding = set(harness.profile_snapshot()["interests"])
    print(f"  After coding msgs: {interests_after_coding}")

    # The new interests should be a SUPERSET of the old ones
    # (gaming interests should still be there)
    gaming_survived = interests_after_gaming.issubset(interests_after_coding)
    has_coding = any(i in interests_after_coding for i in ["coding", "programming"])
    print(f"  Gaming interests survived: {gaming_survived}")
    print(f"  Coding interests added: {has_coding}")

    return gaming_survived and has_coding


def test_axis_smoothing(client):
    """Verify that axes don't flip-flop on every message."""
    print(f"\n{'=' * 60}")
    print("  TEST: Axis Smoothing")
    print(f"{'=' * 60}")

    harness = DynamicChatHarness(client, profile_path=os.devnull)

    # Send several casual messages to establish formality=casual
    casual_msgs = [
        "lol whats good bro",
        "haha yeah cuh thats wild",
        "lmao fr fr no cap",
        "idk man thats tough lol",
    ]
    for msg in casual_msgs:
        harness.send_chat_message(msg)

    profile_before_formal = harness.profile_snapshot()
    formality_before = profile_before_formal["style_axes"]["formality"]
    print(f"  After 4 casual msgs: formality = {formality_before}")

    # Now send ONE formal message — should NOT flip the axis
    harness.send_chat_message("I would like to formally request additional information please.")
    profile_after_formal = harness.profile_snapshot()
    formality_after = profile_after_formal["style_axes"]["formality"]
    print(f"  After 1 formal msg:  formality = {formality_after}")

    # Formality should stay casual (smoothing should absorb the outlier)
    stable = formality_after == "casual"
    print(f"  Axis remained stable: {stable}")
    return stable


def test_prompt_compression(client):
    """Verify that the system prompt injection is compact."""
    print(f"\n{'=' * 60}")
    print("  TEST: Prompt Compression")
    print(f"{'=' * 60}")

    harness = DynamicChatHarness(client, profile_path=os.devnull)

    # Warm up the profile
    for msg in ["hey whats up", "lol nothing much", "haha same"]:
        harness.send_chat_message(msg)
    if harness._profiler_thread and harness._profiler_thread.is_alive():
        harness._profiler_thread.join(timeout=10)

    prompt = harness.assemble_system_prompt()
    line_count = len(prompt.strip().split("\n"))
    char_count = len(prompt)
    print(f"  System prompt: {line_count} lines, {char_count} chars")
    print(f"  Content:\n    " + prompt.replace("\n", "\n    "))

    # Should be <=5 lines
    compact = line_count <= 5
    print(f"  Compact ({line_count} <= 5 lines): {compact}")
    return compact


def test_persistence(client):
    """Verify that profiles survive save/load."""
    print(f"\n{'=' * 60}")
    print("  TEST: Profile Persistence")
    print(f"{'=' * 60}")

    # Use a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    tmp_path = tmp.name
    tmp.close()

    try:
        # Session 1: build a profile
        harness1 = DynamicChatHarness(client, profile_path=tmp_path)
        msgs = [
            "lol gaming is life bro",
            "yeah cuh I play fps games all day",
            "haha the keyboard switches are so satisfying",
        ]
        for msg in msgs:
            harness1.send_chat_message(msg)
        if harness1._profiler_thread and harness1._profiler_thread.is_alive():
            harness1._profiler_thread.join(timeout=10)

        profile1 = harness1.profile_snapshot()
        print(f"  Session 1 profile: style={profile1['style_code']}, interests={profile1['interests']}, slang={profile1['slang']}")

        # Session 2: load the saved profile
        harness2 = DynamicChatHarness(client, profile_path=tmp_path)
        profile2 = harness2.profile_snapshot()
        print(f"  Session 2 loaded:  style={profile2['style_code']}, interests={profile2['interests']}, slang={profile2['slang']}")

        # Check that the key fields survived
        interests_match = set(profile1["interests"]) == set(profile2["interests"])
        slang_match = set(profile1["slang"]) == set(profile2["slang"])
        style_match = profile1["style_code"] == profile2["style_code"]

        print(f"  Interests survived: {interests_match}")
        print(f"  Slang survived:     {slang_match}")
        print(f"  Style survived:     {style_match}")

        return interests_match and slang_match and style_match
    finally:
        os.unlink(tmp_path)


def test_confidence_scores(client):
    """Verify that confidence scores increase with observations."""
    print(f"\n{'=' * 60}")
    print("  TEST: Confidence Scores")
    print(f"{'=' * 60}")

    harness = DynamicChatHarness(client, profile_path=os.devnull)

    conf_before = harness.profile_snapshot()["confidence"]
    print(f"  Before any msgs: {conf_before}")

    harness.send_chat_message("hey whats up lol")
    conf_after_1 = harness.profile_snapshot()["confidence"]
    print(f"  After 1 msg:     {conf_after_1}")

    for msg in ["haha nice", "lmao fr fr", "btw I was thinking"]:
        harness.send_chat_message(msg)
    conf_after_4 = harness.profile_snapshot()["confidence"]
    print(f"  After 4 msgs:    {conf_after_4}")

    # len and form should have observations (heuristic axes)
    # cog and eng should have 0 or 1 (only from profiler, which runs at msg 3)
    len_grows = conf_after_4["len"] > conf_before["len"]
    form_grows = conf_after_4["form"] > conf_before["form"]
    print(f"  len confidence grew: {len_grows}")
    print(f"  form confidence grew: {form_grows}")

    return len_grows and form_grows


def main():
    # Check server
    online, result = check_server_status()
    if not online:
        print(f"[FAIL] Server not reachable: {result}")
        print("   Start the mock server first: python mock_server.py")
        sys.exit(1)

    print(f"[OK] Connected to model: {result}")
    client = LMStudioClient(result)  # No spinner for automated testing

    # Run all test conversations
    results = {}
    results["casual_gamer"], _ = run_conversation(
        "Casual Gamer (slang-heavy, gaming/keyboards)",
        CASUAL_GAMER_CONVERSATION, client,
        profile_path=os.devnull,
    )
    results["formal_student"], _ = run_conversation(
        "Formal Student (proper grammar, studying/grades)",
        FORMAL_STUDENT_CONVERSATION, client,
        profile_path=os.devnull,
    )
    results["mixed_coder"], _ = run_conversation(
        "Mixed Coder (coding + fitness, emoji user)",
        MIXED_CODER_CONVERSATION, client,
        profile_path=os.devnull,
    )

    # Run targeted tests
    accum_pass = test_interest_accumulation(client)
    smooth_pass = test_axis_smoothing(client)
    prompt_pass = test_prompt_compression(client)
    persist_pass = test_persistence(client)
    confid_pass = test_confidence_scores(client)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")

    all_pass = True
    checks = [
        # Original checks
        ("casual_gamer style should be informal (I)", results["casual_gamer"]["style_axes"]["formality"] == "casual"),
        ("formal_student style should be formal (F)", results["formal_student"]["style_axes"]["formality"] == "formal"),
        ("casual_gamer should detect slang", len(results["casual_gamer"]["slang"]) > 0),
        ("formal_student should not detect slang", len(results["formal_student"]["slang"]) == 0),
        ("mixed_coder should detect emoji usage", results["mixed_coder"]["uses_emojis"]),
        ("casual_gamer should find gaming-related interests", any(i in results["casual_gamer"]["interests"] for i in ["gaming", "keyboards"])),
        ("formal_student should find study interests", any(i in results["formal_student"]["interests"] for i in ["studying", "grades", "exams", "school"])),
        ("casual_gamer should have ad suggestions", len(results["casual_gamer"]["potential_sponsored_suggestions"]) > 0),
        # New Phase 1 checks
        ("interest accumulation works", accum_pass),
        ("axis smoothing prevents flip-flop", smooth_pass),
        ("system prompt is compact (<= 5 lines)", prompt_pass),
        ("profile persistence works", persist_pass),
        ("confidence scores increase", confid_pass),
    ]

    for desc, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_pass = False

    print(f"\n  {'All checks passed!' if all_pass else 'Some checks failed'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
