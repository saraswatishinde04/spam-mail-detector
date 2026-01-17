import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Spam Mail Detector", page_icon="📧")

st.title("📧 Spam Mail Detector")
st.write("Enter a message to check whether it is Spam or Not Spam.")

# Text input
message = st.text_area("Enter your message here:")

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        clean_message = preprocess_text(message)
        vectorized_message = vectorizer.transform([clean_message])
        prediction = model.predict(vectorized_message)[0]
        probability = model.predict_proba(vectorized_message)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 Spam Message ({probability*100:.2f}% confidence)")
        else:
            st.success(f"✅ Not Spam ({probability*100:.2f}% confidence)")
