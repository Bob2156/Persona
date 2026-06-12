"""
Profiling utilities — heuristic analysis and LLM-based deep profiling.

These are stateless functions used by the Persona engine.
"""

import re

from persona.defaults import CASUAL_SLANG_PATTERN, EMOJI_PATTERN, NOISE_INTERESTS


def compute_heuristics(last_user_messages):
    """Fast, local heuristics computed on every user message.

    Returns (len_letter, form_letter, uses_emojis) or None if no messages.
    """
    message_count = len(last_user_messages)
    if message_count == 0:
        return None
    recent_text = " ".join(msg["content"] for msg in last_user_messages)
    avg_len = len(recent_text) / message_count
    len_letter = "C" if avg_len < 60 else "V"
    form_letter = "I" if CASUAL_SLANG_PATTERN.search(recent_text.lower()) else "F"
    uses_emojis = bool(EMOJI_PATTERN.search(recent_text))
    return len_letter, form_letter, uses_emojis


def parse_profiler_response(response_text):
    """Parse the structured output from the LLM profiler.

    Returns dict with keys: cog, eng, interests, slang.
    """
    cog_match = re.search(r"COGNITIVE:\s*([ESes])", response_text)
    eng_match = re.search(r"ENGAGEMENT:\s*([PDpd])", response_text)
    int_match = re.search(r"INTERESTS:\s*([^\n]+)", response_text)
    slang_match = re.search(r"SLANG:\s*([^\n]+)", response_text)

    interests = []
    if int_match:
        raw = [item.strip().lower() for item in int_match.group(1).split(",")]
        interests = [item for item in raw if item and item not in NOISE_INTERESTS]

    slang = []
    if slang_match:
        raw = [item.strip().lower() for item in slang_match.group(1).split(",")]
        slang = [item for item in raw if item and item != "none"]

    return {
        "cog": cog_match.group(1).upper() if cog_match else None,
        "eng": eng_match.group(1).upper() if eng_match else None,
        "interests": interests,
        "slang": slang,
    }


def format_transcript(conversation_history, recent_count=6):
    """Format recent conversation history for the profiler."""
    recent = conversation_history[-recent_count:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Chatbot"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def build_profiler_system_prompt():
    """Build the system prompt for the background profiler LLM call."""
    return (
        "You are a background psychometrics engine.\n"
        "Analyze the transcript and output EXACTLY four lines:\n"
        "COGNITIVE: [E if user values emotion/relationships, S if they focus on facts/logic]\n"
        "ENGAGEMENT: [P if user wants open conversation, D if they want direct answers]\n"
        "INTERESTS: [1-3 concrete topic tags like 'gaming', 'coding', 'dating'. Not emotions. Write 'none' if none]\n"
        "SLANG: [Specific slang words the user used, e.g. 'cuh', 'fr fr', 'lol'. Write 'none' if none]\n"
        "Output only these four lines."
    )
