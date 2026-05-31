import re

from config import CASUAL_SLANG_PATTERN, EMOJI_PATTERN, NOISE_INTERESTS


def compute_heuristics(last_user_messages):
    message_count = len(last_user_messages)
    if message_count == 0:
        return None
    recent_text = " ".join(message["content"] for message in last_user_messages)
    avg_len = len(recent_text) / message_count
    len_letter = "C" if avg_len < 60 else "V"
    form_letter = "I" if CASUAL_SLANG_PATTERN.search(recent_text.lower()) else "F"
    uses_emojis = bool(EMOJI_PATTERN.search(recent_text))
    return len_letter, form_letter, uses_emojis


def parse_profiler_response(response_text):
    cog_match = re.search(r"COGNITIVE:\s*([ESes])", response_text)
    eng_match = re.search(r"ENGAGEMENT:\s*([PDpd])", response_text)
    int_match = re.search(r"INTERESTS:\s*([^\n]+)", response_text)
    slang_match = re.search(r"SLANG:\s*([^\n]+)", response_text)

    interests = []
    if int_match:
        raw_interests = [item.strip().lower() for item in int_match.group(1).split(",")]
        interests = [item for item in raw_interests if item and item not in NOISE_INTERESTS]

    slang = []
    if slang_match:
        raw_slang = [item.strip().lower() for item in slang_match.group(1).split(",")]
        slang = [item for item in raw_slang if item and item != "none"]

    return {
        "cog": cog_match.group(1).upper() if cog_match else None,
        "eng": eng_match.group(1).upper() if eng_match else None,
        "interests": interests,
        "slang": slang,
    }


def format_transcript(conversation_history, recent_count=6):
    recent_history = conversation_history[-recent_count:]
    formatted_transcript = ""
    for message in recent_history:
        role = "User" if message["role"] == "user" else "Chatbot"
        formatted_transcript += f"{role}: {message['content']}\n"
    return formatted_transcript


def build_profiler_system_prompt():
    return (
        "You are a background cognitive psychometrics engine.\n"
        "Your task is to analyze the conversation transcript and output EXACTLY four lines:\n"
        "COGNITIVE: [Write 'E' if the user values emotional validation/relationships, or 'S' if they focus on specs/facts/logic]\n"
        "ENGAGEMENT: [Write 'P' if the user responds well to open questions and wants to keep talking, or 'D' if they want a direct answer/no questions]\n"
        "INTERESTS: [List 1 to 3 key topic tags representing sustained commercial, hobby, study, or project interests. Do NOT extract temporary conversational reactions, emotional expressions, or jokes like 'humor', 'sadness', 'validation', or 'clarity'. Only extract concrete subjects like 'gaming', 'grades', 'dating', 'coding'. If none are found, write 'none']\n"
        "SLANG: [List any specific informal/slang words the user has actually used in the transcript in lowercase, e.g. 'cuh', 'blud', 'fr fr', 'sike', 'lol', 'lmao'. If none, write 'none']\n"
        "Output only these four lines and nothing else."
    )
