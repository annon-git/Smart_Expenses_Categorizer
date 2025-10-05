import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load the dataset
data = pd.read_csv('expense_data.csv')
print(data.head())

# Check and drop missing labels
print("Rows with missing labels:", data['category'].isnull().sum())
data = data.dropna(subset=['category'])

# Clean the text
def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove special characters and digits
    return text

data['cleaned_text'] = data['text'].apply(clean_text)
print(data[['text', 'cleaned_text']].head())

# Vectorize the cleaned text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_text'])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Define labels
y = data['category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict on new expense text examples
def predict_category(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

# Example usage
sample_expenses = [
    "Zomato 340",
    "Ola 120",
    "Big Bazaar 4000",
    "Electricity bill 2300",
    "Spotify 149"
]

for exp in sample_expenses:
    print(f"{exp} --> {predict_category(exp)}")

import joblib
joblib.dump(model, "expense_model.pkl")
joblib.dump(vectorizer, "expense_vectorizer.pkl")

# Function for predicting a category
def predict_category(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

# Loop to accept user input until 'exit' is typed
while True:
    user_input = input("Enter a transaction text (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    category = predict_category(user_input)
    print(f"Predicted Category: {category}")

def predict_category(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)
    return prediction[0]

while True:
    sample = input("Enter a transaction (or type 'exit' to quit): ")
    if sample.lower() == "exit":
        break
    print("Predicted Category:", predict_category(sample))

import joblib

# Save the trained model
joblib.dump(model, 'expense_model.pkl')
# Save the vectorizer
joblib.dump(vectorizer, 'expense_vectorizer.pkl')

print("Model and vectorizer saved!")
