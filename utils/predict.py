"""
predict.py
----------
Loads the trained BiLSTM and BERT models (if available) and exposes a
single `predict_emotion()` function used by the Streamlit app.

Design notes:
- Models are loaded lazily and cached, so a missing model (e.g. you
  haven't trained BERT yet) doesn't crash the whole app -- that model's
  results are simply skipped with a friendly message.
- Keyword-based rule enhancement is blended in to make outputs more
  robust on short / ambiguous inputs.
- "Mixed emotions" = any emotion whose probability is within
  MIXED_THRESHOLD of the top probability.
"""

import os
import pickle

import numpy as np

from utils.preprocess import EMOTIONS, clean_text, get_keyword_scores

MODEL_DIR = "models"
MAX_LEN = 40
MIXED_THRESHOLD = 0.15
KEYWORD_WEIGHT = 0.2  # how much the rule-based keyword score influences the final probs

_bilstm_cache = {}
_bert_cache = {}


# ---------------------------------------------------------------------------
# BiLSTM
# ---------------------------------------------------------------------------
def _load_bilstm():
    if "model" in _bilstm_cache:
        return _bilstm_cache

    model_path = os.path.join(MODEL_DIR, "bilstm_model.h5")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if not (os.path.exists(model_path) and os.path.exists(tokenizer_path)
            and os.path.exists(encoder_path)):
        return None

    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    _bilstm_cache["model"] = model
    _bilstm_cache["tokenizer"] = tokenizer
    _bilstm_cache["label_encoder"] = label_encoder
    return _bilstm_cache


def predict_bilstm(text: str):
    """Returns dict {emotion: probability} or None if model unavailable."""
    cache = _load_bilstm()
    if cache is None:
        return None

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text(text)
    seq = cache["tokenizer"].texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = cache["model"].predict(padded, verbose=0)[0]

    classes = cache["label_encoder"].classes_
    return {cls: float(p) for cls, p in zip(classes, probs)}


# ---------------------------------------------------------------------------
# BERT
# ---------------------------------------------------------------------------
def _load_bert():
    if "model" in _bert_cache:
        return _bert_cache

    model_path = os.path.join(MODEL_DIR, "bert_model")
    encoder_path = os.path.join(MODEL_DIR, "bert_label_encoder.pkl")

    if not (os.path.isdir(model_path) and os.path.exists(encoder_path)):
        return None

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    _bert_cache["model"] = model
    _bert_cache["tokenizer"] = tokenizer
    _bert_cache["label_encoder"] = label_encoder
    return _bert_cache


def predict_bert(text: str):
    """Returns dict {emotion: probability} or None if model unavailable."""
    cache = _load_bert()
    if cache is None:
        return None

    import torch

    cleaned = clean_text(text)
    inputs = cache["tokenizer"](
        cleaned, return_tensors="pt", truncation=True, padding=True, max_length=48
    )

    with torch.no_grad():
        logits = cache["model"](**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]

    classes = cache["label_encoder"].classes_
    return {cls: float(p) for cls, p in zip(classes, probs)}


# ---------------------------------------------------------------------------
# Combination logic
# ---------------------------------------------------------------------------
def _blend_with_keywords(model_probs: dict, text: str) -> dict:
    keyword_scores = get_keyword_scores(text)
    blended = {}
    for emotion in EMOTIONS:
        m = model_probs.get(emotion, 0.0)
        k = keyword_scores.get(emotion, 0.0)
        blended[emotion] = (1 - KEYWORD_WEIGHT) * m + KEYWORD_WEIGHT * k

    total = sum(blended.values())
    if total > 0:
        blended = {k_: v / total for k_, v in blended.items()}
    return blended


def get_mixed_emotions(probs: dict, threshold: float = MIXED_THRESHOLD):
    """Returns list of (emotion, prob) sorted desc, for emotions within
    `threshold` of the top probability -- i.e. the 'mixed emotion' set."""
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_prob = sorted_probs[0][1]
    mixed = [(e, p) for e, p in sorted_probs if (top_prob - p) <= threshold]
    return mixed


def predict_emotion(text: str, use_bilstm=True, use_bert=True):
    """
    Main entry point used by the Streamlit app.

    Returns a dict:
    {
        "bilstm": {emotion: prob} or None,
        "bert": {emotion: prob} or None,
        "combined": {emotion: prob},   # blended + keyword-enhanced
        "top_emotion": str,
        "confidence": float,
        "mixed_emotions": [(emotion, prob), ...],
    }
    """
    bilstm_probs = predict_bilstm(text) if use_bilstm else None
    bert_probs = predict_bert(text) if use_bert else None

    available = [p for p in (bilstm_probs, bert_probs) if p is not None]

    if not available:
        # Fall back to pure keyword-based scoring so the app never crashes
        combined = get_keyword_scores(text)
    else:
        avg = {e: np.mean([p[e] for p in available]) for e in EMOTIONS}
        combined = _blend_with_keywords(avg, text)

    top_emotion = max(combined, key=combined.get)
    confidence = combined[top_emotion]
    mixed = get_mixed_emotions(combined)

    return {
        "bilstm": bilstm_probs,
        "bert": bert_probs,
        "combined": combined,
        "top_emotion": top_emotion,
        "confidence": confidence,
        "mixed_emotions": mixed,
    }
