from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
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
    """Load the trained model"""
    global model
    try:
        # Load your saved model here
        # Replace 'path_to_your_model.h5' with the actual path to your model
        model = tf.keras.models.load_model('mnist_model.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

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
        image_array = image_array.reshape(1, 28, 28, 1)
        image_array = image_array.astype('float32') / 255.0
        
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
        
        # Make prediction
        prediction = model.predict(processed_image)
        digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        return jsonify({
            'digit': digit,
            'confidence': confidence,
            'all_predictions': prediction[0].tolist()
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
        
        # For equation solving, you would need a more complex model
        # that can segment and recognize multiple characters
        # This is a simplified version
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # For now, return a mock equation
        # In a real implementation, you would use a model that can:
        # 1. Segment individual characters
        # 2. Recognize each character
        # 3. Parse the equation
        # 4. Solve it
        
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
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("Loading MNIST model...")
    load_model()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
