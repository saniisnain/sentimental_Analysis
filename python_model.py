from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained sentiment analysis model
model = joblib.load('sentimental_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return "Sentiment Analysis API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON request
    data = request.get_json(force=True)

    # 'text' should be the input key containing the sentence or document
    try:
        input_text = [data['text']]  # wrap in list to make it iterable for vectorizers/models
    except Exception as e:
        return jsonify({'error': 'Invalid input format. Expected "text": "<your sentence here>"', 'details': str(e)}), 400

    # Make prediction
    prediction = model.predict(input_text)[0]

    # Optional: map numeric prediction to label
    sentiment = 'positive' if prediction == 1 else 'negative'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
