"""
Default configuration for the Persona engine.

All values can be overridden via the Persona constructor.
"""

import re

# -- Profiling engine tuning --
SMOOTHING_WINDOW = 5            # recent observations per axis for majority-vote
CONFIDENCE_THRESHOLD = 3        # min observations before injecting style rules
MAX_INTERESTS = 15              # cap on accumulated interest tags
MAX_SLANG = 8                   # cap on tracked slang terms
PROFILER_INTERVAL = 3           # run deep profiler every N user messages
PROFILER_TRANSCRIPT_WINDOW = 6  # how many recent messages the profiler sees

# -- Style rules: maps axis letters to compact behavioral directives --
STYLE_RULES = {
    "C": "Keep replies short (1-2 sentences).",
    "V": "Give detailed but conversational answers.",
    "S": "Be direct and logical, skip emotional fluff.",
    "E": "Show friendly warmth and acknowledge their mood.",
    "F": "Use clean, proper grammar.",
    "I": "Keep it casual, match their vibe.",
    "D": "Answer directly, no forced follow-up questions.",
    "P": "End with a natural open-ended question.",
}

# -- Heuristic detection patterns --
CASUAL_SLANG_PATTERN = re.compile(
    r"\blol\b|\bbtw\b|\bhaha\b|\bgonna\b|\bcuh\b|\blmao\b|\bidk\b"
    r"|\bwazzup\b|\bblud\b|\bfoo\b|\bfr\s+fr\b|\bbruh\b|\bngl\b"
    r"|\bimo\b|\bsmh\b|\bfam\b|\bno\s+cap\b|\bbet\b|\bsus\b"
)
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]")

# -- Interests that are conversational noise, not real topics --
NOISE_INTERESTS = {"none", "humor", "sadness", "validation", "clarity", "crisis"}
