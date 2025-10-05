from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('expense_model.pkl')
vectorizer = joblib.load('expense_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    
    return text

def predict_category(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    category = predict_category(text)
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True)
