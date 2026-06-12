import re

API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODELS_URL = "http://127.0.0.1:1234/v1/models"
SHOW_PROFILE_DEBUG = False

# -- Profiling engine tuning --
SMOOTHING_WINDOW = 5            # number of recent observations to consider per axis
CONFIDENCE_THRESHOLD = 3        # min observations before an axis is considered "confident"
MAX_INTERESTS = 10              # cap on accumulated interest tags
PROFILER_INTERVAL = 3           # run deep profiler every N user messages
PROFILE_SAVE_PATH = "persona_profile.json"  # default save location

AD_MAPPING = {
    "grades": "AI-powered flashcard & study planner licenses",
    "school": "AI-powered flashcard & study planner licenses",
    "studying": "AI-powered flashcard & study planner licenses",
    "exams": "AI-powered flashcard & study planner licenses",
    "gaming": "ergonomic hot-swappable mechanical keyboards",
    "keyboards": "ergonomic hot-swappable mechanical keyboards",
    "dating": "premium algorithmic matchmaking subscriptions",
    "relationships": "premium algorithmic matchmaking subscriptions",
    "coding": "advanced local copilot licenses",
    "programming": "advanced local copilot licenses",
    "fitness": "orthopedic high-performance running shoes",
    "running": "orthopedic high-performance running shoes",
}

MBTI_RULES = {
    "C": "Keep your response very short, conversational, and to the point (1 to 2 sentences max).",
    "V": "Provide a naturally detailed response, but keep it conversational and avoid writing long essays.",
    "S": "Be direct, matter-of-fact, and logical. Skip any emotional validation or warm conversational fluff.",
    "E": "Show friendly interest and natural warmth. Acknowledge the user's mood naturally, without sounding like a therapist.",
    "F": "Write with clean, polite, and proper grammar. Keep your tone professional.",
    "I": "Keep your tone highly casual and friendly, matching the user's conversational flow.",
    "D": "Answer directly and do not force extra follow-up questions at the end.",
    "P": "End with a natural open-ended question that keeps the conversation going.",
}

NOISE_INTERESTS = {"none", "humor", "sadness", "validation", "clarity", "crisis"}
CASUAL_SLANG_PATTERN = re.compile(
    r"\blol\b|\bbtw\b|\bhaha\b|\bgonna\b|\bcuh\b|\blmao\b|\bidk\b|\bwazzup\b|\bblud\b|\bfoo\b|\bfr\s+fr\b"
)
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]")
