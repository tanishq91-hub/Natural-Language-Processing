import streamlit as st
import joblib

# -----------------------------
# Load saved files
# -----------------------------
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder (1).pkl")  # this is a dict

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("🎭 Emotion Detection App")
st.write("Predict emotion: Happy, Sad, Joy, Fear, Angry")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize text
        text_vectorized = vectorizer.transform([user_input])

        # Predict class index
        prediction = model.predict(text_vectorized)[0]

        # Convert index → label using dict
        final_result = label_encoder[prediction]

        st.success(f"Prediction: **{final_result}**")
