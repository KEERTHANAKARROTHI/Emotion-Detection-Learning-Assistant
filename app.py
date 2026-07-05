"""
app.py
------
AI-Driven Emotion Detection & Personalized Learning Support Platform.

Run:
    streamlit run app.py
"""

import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.predict import predict_emotion, EMOTIONS
from utils.gemini import generate_response

LOG_PATH = os.path.join("logs", "interactions.csv")
os.makedirs("logs", exist_ok=True)

st.set_page_config(
    page_title="AI Emotion-Aware Learning Assistant",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log_interaction(text, emotion, confidence, mixed_emotions, ai_response):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text": text,
        "top_emotion": emotion,
        "confidence": round(confidence, 4),
        "mixed_emotions": ", ".join([f"{e}({p:.2f})" for e, p in mixed_emotions]),
        "ai_response": ai_response if ai_response else "",
    }
    df_row = pd.DataFrame([row])

    if os.path.exists(LOG_PATH):
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode="w", header=True, index=False)


def load_logs():
    if os.path.exists(LOG_PATH):
        try:
            return pd.read_csv(LOG_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def plot_probs_bar(probs: dict, title: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    emotions = list(probs.keys())
    values = [probs[e] for e in emotions]
    colors = plt.cm.viridis([v for v in values])
    ax.bar(emotions, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

api_key_input = st.sidebar.text_input(
    "Gemini API Key", type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Get one at https://aistudio.google.com/apikey",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model comparison")
use_bilstm = st.sidebar.checkbox("Use BiLSTM", value=True)
use_bert = st.sidebar.checkbox("Use BERT", value=True)

show_ai_response = st.sidebar.toggle("Generate AI guidance (Gemini)", value=True)

st.sidebar.markdown("---")
mode = st.sidebar.radio("View", ["🏠 Assistant", "📊 Analytics Dashboard"])

# ---------------------------------------------------------------------------
# Main: Assistant view
# ---------------------------------------------------------------------------
if mode == "🏠 Assistant":
    st.title("🎓 AI-Driven Emotion Detection & Learning Support")
    st.caption(
        "Describe your study challenge below. We'll detect your emotional "
        "state and give you tailored, supportive guidance."
    )

    text_input = st.text_area(
        "What are you struggling with (or excited about) today?",
        placeholder="e.g. I'm lost on recursion and don't know where to start...",
        height=120,
    )

    analyze_clicked = st.button("Analyze", type="primary")

    if analyze_clicked:
        if not text_input.strip():
            st.warning("Please enter some text describing your challenge.")
        else:
            with st.spinner("Analyzing emotion..."):
                result = predict_emotion(
                    text_input, use_bilstm=use_bilstm, use_bert=use_bert
                )

            top_emotion = result["top_emotion"]
            confidence = result["confidence"]
            mixed = result["mixed_emotions"]

            st.subheader(f"Detected Emotion: **{top_emotion}** ({confidence:.0%} confidence)")

            if len(mixed) > 1:
                mixed_str = " + ".join([f"{e} ({p:.0%})" for e, p in mixed])
                st.info(f"🌀 Mixed emotions detected: {mixed_str}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Combined Prediction**")
                st.pyplot(plot_probs_bar(result["combined"], "Combined + Keyword-Enhanced"))

            with col2:
                st.markdown("**BiLSTM Prediction**")
                if result["bilstm"]:
                    st.pyplot(plot_probs_bar(result["bilstm"], "BiLSTM"))
                else:
                    st.warning("BiLSTM model not found. Train it with `python train_bilstm.py`.")

            with col3:
                st.markdown("**BERT Prediction**")
                if result["bert"]:
                    st.pyplot(plot_probs_bar(result["bert"], "BERT"))
                else:
                    st.warning("BERT model not found. Train it with `python train_bert.py`.")

            ai_response = None
            if show_ai_response:
                st.markdown("---")
                st.subheader("💡 Personalized Guidance")
                with st.spinner("Generating supportive response..."):
                    ai_response = generate_response(
                        text_input, top_emotion, confidence, api_key=api_key_input
                    )
                st.write(ai_response)

            log_interaction(text_input, top_emotion, confidence, mixed, ai_response)
            st.success("Interaction logged for progress tracking.")

# ---------------------------------------------------------------------------
# Main: Analytics dashboard
# ---------------------------------------------------------------------------
else:
    st.title("📊 Emotion Analytics Dashboard")

    logs = load_logs()

    if logs.empty:
        st.info("No interactions logged yet. Use the Assistant tab first.")
    else:
        st.subheader("Recent Interactions")
        st.dataframe(logs.tail(20), use_container_width=True)

        st.subheader("Emotion Distribution")
        counts = logs["top_emotion"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index, counts.values, color="skyblue")
        ax.set_ylabel("Count")
        plt.xticks(rotation=20)
        st.pyplot(fig)

        st.subheader("Average Confidence by Emotion")
        avg_conf = logs.groupby("top_emotion")["confidence"].mean()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(avg_conf.index, avg_conf.values, color="salmon")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Avg Confidence")
        plt.xticks(rotation=20)
        st.pyplot(fig2)

        csv_bytes = logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full interaction log (CSV)",
            data=csv_bytes,
            file_name="interactions.csv",
            mime="text/csv",
        )