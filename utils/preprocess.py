"""
preprocess.py
-------------
Shared text-cleaning and keyword-based rule-enhancement utilities used by
both the training scripts and the prediction pipeline.
"""

import re

EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

# Keyword lists used for rule-based enhancement of model predictions.
KEYWORDS = {
    "Bored": [
        "bored", "boring", "tedious", "dull", "monotonous", "sleepy",
        "uninteresting", "yawn",
    ],
    "Confident": [
        "confident", "sure", "easy", "got this", "comfortable", "capable",
        "understand it well", "i can do this", "no problem",
    ],
    "Confused": [
        "confused", "lost", "don't understand", "dont understand", "unclear",
        "stuck", "puzzled", "not sure how", "no idea", "makes no sense",
    ],
    "Curious": [
        "curious", "interesting", "wonder", "want to know", "fascinated",
        "intrigued", "explore", "what if", "how does",
    ],
    "Frustrated": [
        "frustrated", "annoyed", "angry", "give up", "hate this",
        "so hard", "frustrating", "driving me crazy", "lose my patience",
    ],
}


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace for model input."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_keyword_scores(text: str) -> dict:
    """
    Returns a normalized dict {emotion: score in [0,1]} based on
    keyword occurrence counts in the raw (lowercased) text.
    """
    raw = text.lower()
    scores = {emotion: 0.0 for emotion in EMOTIONS}

    for emotion, words in KEYWORDS.items():
        count = 0
        for kw in words:
            if kw in raw:
                count += 1
        scores[emotion] = float(count)

    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    else:
        # No keyword matches -> uniform, contributes neutrally
        scores = {k: 1.0 / len(EMOTIONS) for k in EMOTIONS}

    return scores
