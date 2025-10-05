import streamlit as st
import joblib

model = joblib.load("expense_model.pkl")
vectorizer = joblib.load("expense_vectorizer.pkl")

def clean_text(text):
    return text.lower()  # Use your real clean_text from training

st.title("Smart Expense Categorizer")

user_input = st.text_input("Enter an expense (e.g., Swiggy 340)")

if user_input:
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    st.success(f"Predicted Category: {result}")
