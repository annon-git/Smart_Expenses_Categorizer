Smart Expense Categorizer (ML)
An AI agent that automates classification of SMS/transaction texts and suggests daily study plans for expense management.

Project Objective
This project classifies transaction texts (e.g., 'Swiggy 250' → Food, 'Uber 300' → Travel) using machine learning. It also demonstrates a daily study plan suggestion using AI.

Tech Stack
Python 3.x
scikit-learn (TF-IDF, Logistic Regression)
Pandas, NumPy
Streamlit for UI

Implementation Details
Dataset: Labeled SMS/transaction samples (Food, Travel, etc.)
Model Training: Used TF-IDF + Logistic Regression for classification
App: Includes both CLI and UI app for demonstration

Directory Structure
app.py – Main application file
expense_classifier.py – Training and classification logic
expense_data.csv – Labeled data
expense_model.pkl – Trained machine learning model
expense_vectorizer.pkl – TF-IDF vectorizer
test_api.py – Script to test endpoints/integration
ui_app.py – User interface



images/ – Screenshots of the project
