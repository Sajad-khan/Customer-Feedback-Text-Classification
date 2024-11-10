from flask import Flask, request, jsonify
import joblib
from content.data_preprocessing import load_and_preprocess_data

# Load model and encoder
pipeline = joblib.load('content/model/customer_feedback_classifier.pkl')
label_encoder = joblib.load('content/model/label_encoder.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    feedback_text = data['feedback']

    # Preprocess and predict
    processed_text = feedback_text.lower()
    prediction = pipeline.predict([processed_text])[0]
    sentiment = label_encoder.inverse_transform([prediction])[0]

    return jsonify({'feedback': feedback_text, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run()
