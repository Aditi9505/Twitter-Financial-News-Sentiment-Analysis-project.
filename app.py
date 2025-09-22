import streamlit as st
import pandas as pd
import re
import string
import joblib
from nltk.corpus import stopwords
import nltk

# --- Pre-requisites: Download NLTK data ---
# This is usually done once. If you've run it in your notebook, you might not need it here.
# However, for deployment environments, it's good practice.
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Load Trained Model and Vectorizer ---
# Load the pre-trained model and vectorizer
try:
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
except FileNotFoundError:
    st.error("Model or vectorizer not found. Please run the notebook to train and save them first.")
    st.stop()

# --- Label Mapping ---
label_mapping = {
    0: "Analyst Update", 1: "Fed / Central Banks", 2: "Company / Product News",
    3: "Treasuries / Corporate Debt", 4: "Dividend", 5: "Earnings",
    6: "Energy / Oil", 7: "Financials", 8: "Currencies",
    9: "General News / Opinion", 10: "Gold / Metals / Materials", 11: "IPO",
    12: "Legal / Regulation", 13: "M&A / Investments", 14: "Macro",
    15: "Markets", 16: "Politics", 17: "Personnel Change",
    18: "Stock Commentary", 19: "Stock Movement"
}

# --- Text Cleaning Function ---
def clean_text(text):
    """Cleans the input text by removing URLs, punctuation, and converting to lowercase."""
    text = text.lower()
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

# --- Streamlit App Interface ---
st.set_page_config(page_title="Financial Tweet Sentiment Analysis", layout="wide")

st.title("📈 Twitter Financial News Sentiment Analysis")
st.markdown("Enter a financial news tweet below to classify its category using a trained Logistic Regression model.")

# --- User Input and Prediction ---
user_input = st.text_area("Enter a financial tweet:", "The stock market is showing bullish signs today as tech stocks rally.")

if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Clean the input text
        clean_input = clean_text(user_input)

        # 2. Vectorize the cleaned text
        input_vector = vectorizer.transform([clean_input]).toarray()

        # 3. Predict using the loaded model
        prediction = model.predict(input_vector)
        predicted_label = prediction[0]

        # 4. Get the category name from the mapping
        category = label_mapping.get(predicted_label, "Unknown Category")

        # 5. Display the result
        st.success(f"**Predicted Category:** {category} (Label: {predicted_label})")
    else:
        st.warning("Please enter a tweet to analyze.")

# --- Footer ---
st.markdown("---")
st.write("Developed based on the analysis from `twitter.ipynb`.")
