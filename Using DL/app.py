import streamlit as st
import numpy as np
import pickle
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K


class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        super().build(input_shape)

    def call(self, inputs):
        score = K.tanh(K.dot(inputs, self.W))
        attention_weights = K.softmax(score, axis=1)
        context = inputs * attention_weights
        return K.sum(context, axis=1)


@st.cache_resource
def load_assets():
    model = load_model(
        "emotion_model.keras",
        custom_objects={"Attention": Attention}
    )
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_assets()
MAX_LEN = 150


st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🧠",
    layout="centered"
)


dark = st.toggle("Dark Mode", value=False)

bg = "#0f172a" if dark else "#f8fafc"
card = "rgba(255,255,255,0.08)" if dark else "#ffffff"
text = "#ffffff" if dark else "#0f172a"
border = "rgba(255,255,255,0.2)" if dark else "rgba(0,0,0,0.15)"


st.markdown(
    f"""
    <style>
    .stApp {{
        background: {bg};
        color: {text};
    }}

    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: {text} !important;
    }}

    .glass {{
        background: {card};
        border-radius: 16px;
        padding: 24px;
        border: 1px solid {border};
    }}

    textarea {{
        background: {card} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
    }}

    button {{
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 10px !important;
    }}

    div[data-baseweb="radio"] label {{
        color: {text} !important;
        font-weight: 500;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    "<h1 style='text-align:center;'>Emotion Detection from Text</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>NLP based emotion classification</p>",
    unsafe_allow_html=True
)


mode = st.radio(
    "Mode",
    ["Single Prediction", "Batch Prediction"],
    horizontal=True
)


emotion_emojis = {
    "joy": "🙂",
    "sadness": "😔",
    "anger": "😠",
    "fear": "😨",
    "love": "❤",
    "surprise": "😮"
}


def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = model.predict(pad, verbose=0)[0]
    emotion = label_encoder.inverse_transform([np.argmax(probs)])[0]
    return emotion, probs


if mode == "Single Prediction":
    examples = [
        "I feel very happy and excited today",
        "I am scared and anxious about my future",
        "I feel lonely and worthless",
        "I am so angry and frustrated",
        "I feel loved and cared for"
    ]

    if "predicted" not in st.session_state:
        st.session_state.predicted = False

    left, right = st.columns([1, 1])

    with left:
        selected = st.radio("Examples", examples)
        text = st.text_area("Text", value=selected, height=150)

        if st.button("Predict"):
            emotion, probs = predict(text)
            st.session_state.predicted = True
            st.session_state.emotion = emotion
            st.session_state.probs = probs

            confidence = np.max(probs) * 100
            emoji = emotion_emojis.get(emotion, "")

            st.markdown(
                f"""
                <div class='glass'>
                    <h3>{emotion.upper()}</h3>
                    <p>Confidence: {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with right:
        if st.session_state.predicted:
            df = pd.DataFrame({
                "Emotion": label_encoder.classes_,
                "Probability": st.session_state.probs
            })
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.bar_chart(df.set_index("Emotion"))
            st.markdown("</div>", unsafe_allow_html=True)


else:
    batch_text = st.text_area(
        "Sentences (one per line)",
        height=200
    )

    if st.button("Run"):
        lines = [l for l in batch_text.split("\n") if l.strip()]
        results = []

        for line in lines:
            emotion, _ = predict(line)
            results.append({
                "Text": line,
                "Emotion": emotion
            })

        df = pd.DataFrame(results)
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>TensorFlow · Keras · Streamlit</p>",
    unsafe_allow_html=True
)
