import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('sentimental_model.pkl')

@app.route('/')
def home():
    return 'Sentiment analysis model is running!'

def predict():
    data = request.get_json(force=True)
    text = data['text']
    prediction = model.predict([text])
    return jsonify({'sentiment': int(prediction[0])})



if __name__ == "__main__":
    # Bind to 0.0.0.0 to make the app accessible externally and use the Railway port
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
