import streamlit as st
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page Config ---
st.set_page_config(page_title="Sentiment Classifier", page_icon="🤖")
st.title("🤖 LLM Comment Classifier")
st.write("Type a comment below and the model will classify it as Negative, Neutral, or Positive.")


# --- Load Model & Tokenizer (Cached so it doesn't reload every time) ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("sentiment_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_assets()
MAX_LEN = 60
LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


# --- Clean Text Function (From your predict.py) ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- UI Input ---
user_input = st.text_area("Enter your comment here:", "This app is absolutely amazing!")

if st.button("Classify"):
    if user_input.strip():
        # Preprocess
        seq = tokenizer.texts_to_sequences([clean_text(user_input)])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

        # Predict
        probs = model.predict(padded, verbose=0)[0]
        idx = int(np.argmax(probs))

        # Display Results
        st.subheader(f"Prediction: **{LABELS[idx]}**")
        st.write(f"Confidence: {probs[idx] * 100:.2f}%")

        # Show probabilities
        st.bar_chart({
            "Negative": probs[0],
            "Neutral": probs[1],
            "Positive": probs[2]
        })
    else:
        st.warning("Please enter a comment first.")