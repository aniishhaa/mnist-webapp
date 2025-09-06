from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import json

app = Flask(__name__)
CORS(app)

# Global variable to store the model
model = None

def load_model():
    """Load the trained PyTorch model"""
    global model
    try:
        # For now, we'll use a mock model
        # Later we'll add the real PyTorch model
        model = "mock_model"
        print("✅ Mock model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image_data):
    """Preprocess image for MNIST model"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 (MNIST input size)
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        
        # Normalize to 0-1 range
        image_array = image_array.astype(np.float32) / 255.0
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    """Predict digit from canvas drawing"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # For now, return a mock prediction
        # This will be replaced with real model prediction later
        digit = np.random.randint(0, 10)
        confidence = np.random.random() * 0.4 + 0.6  # 60-100% confidence
        
        return jsonify({
            'digit': int(digit),
            'confidence': float(confidence),
            'all_predictions': [0.1] * 10  # Mock predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_equation', methods=['POST'])
def predict_equation():
    """Predict equation from canvas drawing"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Mock equation for testing
        mock_equations = [
            '12 + 7', '15 - 8', '6 × 4', '20 ÷ 5', '3 + 4 × 2',
            '10 - 3 + 2', '8 ÷ 2 + 1', '5 × 3 - 7', '9 + 6 ÷ 2'
        ]
        
        equation = mock_equations[np.random.randint(0, len(mock_equations))]
        
        # Calculate result
        try:
            clean_equation = equation.replace('×', '*').replace('÷', '/')
            result = eval(clean_equation)
        except:
            result = 'Error'
        
        return jsonify({
            'equation': equation,
            'result': result,
            'steps': [
                'Raw Input: Handwritten equation detected',
                f'Segmentation: Individual characters isolated',
                f'Recognition: "{equation}"',
                'Parsing: Mathematical expression parsed',
                f'Calculation: {equation} = {result}'
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Flask API is running!'
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📱 Loading mock model...")
    
    if load_model():
        print("✅ Mock model loaded successfully!")
        print("🌐 Starting server on http://localhost:5000")
        print("💡 This is a test version - predictions will be random")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model.")
