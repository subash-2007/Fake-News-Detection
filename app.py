from flask import Flask, request, jsonify
from main import FakeNewsDetector
import os

app = Flask(__name__)
detector = FakeNewsDetector()

# Load the saved model
model_path = 'models/fake_news_model.joblib'
if os.path.exists(model_path):
    detector.load_saved_model(model_path)
else:
    raise FileNotFoundError("Model file not found. Please train the model first.")

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess the text
        processed_text = detector.preprocess_text(data['text'])
        
        # Vectorize the text
        text_vec = detector.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = detector.model.predict(text_vec)[0]
        probability = detector.model.predict_proba(text_vec)[0].max()
        
        return jsonify({
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': float(probability),
            'processed_text': processed_text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 