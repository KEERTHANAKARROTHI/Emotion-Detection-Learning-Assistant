"""
gemini.py
---------
Wrapper around the (legacy but still functional) `google-generativeai`
package for generating emotion-aware, supportive guidance for a student's
study challenge.

NOTE: `google-generativeai` is in Google's "limited maintenance" mode as
of late 2025 -- it still works, but for new projects Google recommends
migrating to the newer `google-genai` package (see README.md). This file
uses `google-generativeai` since that's the package already installed.
"""

import os

import google.generativeai as genai

# Model names that are currently served by the Gemini API, tried in order.
# If one is deprecated/unavailable for your API key, the next is tried.
CANDIDATE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]

_configured = False


def configure_api(api_key: str):
    """Call once with the user's API key before generating responses."""
    global _configured
    genai.configure(api_key=api_key)
    _configured = True


def _get_working_model():
    last_error = None
    for name in CANDIDATE_MODELS:
        try:
            model = genai.GenerativeModel(model_name=name)
            return model
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue
    raise RuntimeError(f"No usable Gemini model found. Last error: {last_error}")


def generate_response(text: str, emotion: str, confidence: float, api_key: str = None):
    """
    Generates a short, empathetic, emotion-aware response with tips and
    next steps for the student's described challenge.

    Returns a plain string. Never raises -- returns a friendly fallback
    message on any failure so the Streamlit app keeps working.
    """
    global _configured

    if api_key and not _configured:
        configure_api(api_key)

    if not _configured:
        return (
            "⚠️ Gemini API key not configured. Please enter your API key in "
            "the sidebar to enable AI-generated guidance."
        )

    prompt = f"""
You are a warm, encouraging academic support assistant.

A student described their study challenge as:
"{text}"

Our emotion detection system classified their emotional state as: {emotion}
(confidence: {confidence:.0%})

Write a short, supportive response (max 120 words) that:
1. Acknowledges how they might be feeling ({emotion}), in an empathetic tone.
2. Gives 2-3 concrete, practical tips or next steps relevant to their challenge.
3. Ends with a brief line of encouragement.

Keep the tone warm, human, and non-robotic. Do not repeat the emotion label
verbatim in a clinical way -- weave it naturally into the response.
"""

    try:
        model = _get_working_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:  # noqa: BLE001
        return (
            "⚠️ Could not reach Gemini right now, so here's a quick fallback tip: "
            "break your problem into smaller steps, revisit one worked example, "
            "and try explaining the concept out loud to yourself. "
            f"(Technical detail: {e})"
        )
