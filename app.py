import streamlit as st
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Sentiment Classifier", page_icon="🤖")
st.title("🤖 LLM Comment Classifier")

# CRITICAL: must match the model's input length (was 40, model expects 60)
MAX_LEN = 60


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("sentiment_model_v2.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


try:
    model, tokenizer = load_assets()
    st.success("Model and Tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


user_input = st.text_area("Enter your comment here:", "This is terrible and I hate it.")

if st.button("Classify"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        st.write(f"**Cleaned Text:** `{cleaned}`")

        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

        probs = model.predict(padded, verbose=0)[0]
        idx = int(np.argmax(probs))

        # --- Prediction summary ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", LABELS[idx])
        with col2:
            st.metric("Confidence", f"{probs[idx] * 100:.2f}%")

        # --- Probability bar chart ---
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame(
            {
                "Sentiment": [LABELS[i] for i in range(len(probs))],
                "Probability": [float(p) for p in probs],
            }
        ).set_index("Sentiment")

        st.bar_chart(prob_df, height=300, use_container_width=True)

        # --- Raw values (collapsed by default) ---
        with st.expander("Raw probabilities"):
            st.json({LABELS[i]: float(probs[i]) for i in range(len(probs))})