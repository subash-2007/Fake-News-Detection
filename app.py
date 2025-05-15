from flask import Flask, render_template, request, jsonify, make_response
from main import FakeNewsDetector
import os
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='template', static_folder='static')

detector = FakeNewsDetector()

# Load the saved model
model_path = 'models/fake_news_model.joblib'
if os.path.exists(model_path):
    logger.info(f"Loading model from: {model_path}")
    detector.load_saved_model(model_path)
    logger.info("Model and vectorizer loaded successfully.")
else:
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

# Add caching for text preprocessing to improve performance
@lru_cache(maxsize=128)
def cached_preprocess(text):
    """Cached version of text preprocessing to improve performance for repeated requests"""
    return detector.preprocess_text(text)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data (support both JSON and form data)
        if request.is_json:
            data = request.get_json()
            logger.debug(f"Received JSON data: {data}")
            if not data or 'text' not in data:
                logger.error("No text provided in JSON request")
                return jsonify({'error': 'No text provided'}), 400
            input_text = data['text']
        else:
            input_text = request.form.get('text')
            logger.debug(f"Received form data: {input_text}")
            if not input_text:
                logger.error("No text provided in form request")
                return jsonify({'error': 'No text provided'}), 400

        # Validate input text
        if len(input_text.strip()) < 10:
            logger.warning("Input text too short")
            return jsonify({
                'error': 'Input text too short',
                'message': 'Please provide a longer text for better analysis'
            }), 400

        # Preprocess the text using cached function
        processed_text = cached_preprocess(input_text)
        logger.debug(f"Processed text: {processed_text[:100]}...")

        # Check if we have meaningful text after preprocessing
        if not processed_text or len(processed_text.split()) < 3:
            logger.warning("Insufficient text content after preprocessing")
            return jsonify({
                'error': 'Insufficient content',
                'message': 'After removing stop words and processing, there is not enough meaningful content to analyze'
            }), 400

        # Vectorize the input
        try:
            text_vec = detector.vectorizer.transform([processed_text])
            logger.debug("Vectorization successful")
        except Exception as e:
            logger.error(f"Vectorization error: {str(e)}")
            return jsonify({'error': 'Failed to process text', 'details': str(e)}), 500

        # Predict
        try:
            prediction = detector.model.predict(text_vec)[0]
            probabilities = detector.model.predict_proba(text_vec)[0]
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            confidence_margin = abs(probabilities[1] - probabilities[0])
            
            logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Margin: {confidence_margin:.4f}")
            
            # Define confidence levels for better UX
            confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            
            return jsonify({
                'prediction': 'Real' if prediction == 1 else 'Fake',
                'confidence': float(confidence),
                'confidence_level': confidence_level,
                'confidence_margin': float(confidence_margin),
                'probabilities': {
                    'fake': float(probabilities[0]),
                    'real': float(probabilities[1])
                },
                'processed_text': processed_text,
                'word_count': len(processed_text.split())
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Failed to make prediction', 'details': str(e)}), 500

    except Exception as e:
        logger.error(f"Unhandled exception during prediction: {str(e)}")
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for external services"""
    response = predict()
    if isinstance(response, tuple):
        return response
    
    # Set CORS headers for API endpoint
    resp = make_response(response)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    if detector.model and detector.vectorizer:
        status = 'healthy'
        code = 200
    else:
        status = 'degraded'
        code = 503
        
    return jsonify({
        'status': status,
        'model_loaded': detector.model is not None,
        'vectorizer_loaded': detector.vectorizer is not None,
    }), code

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true', 
            host='0.0.0.0', 
            port=port) 