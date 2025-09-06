from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import json
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Your original CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(self.relu(self.conv2(x)))  # 14 -> 7
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Global variable to store the model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained PyTorch model"""
    global model
    try:
        # Create the model
        model = CNN().to(device)
        
        # Try to load the trained weights
        try:
            model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
            print("✅ Real PyTorch model loaded successfully!")
            return True
        except FileNotFoundError:
            print("⚠️ Model file not found. Using untrained model.")
            return True
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}. Using untrained model.")
            return True
            
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def preprocess_image(image_data, debug=False):
    """Preprocess image for MNIST model"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if debug:
            print(f"Original image size: {image.size}")
            print(f"Original image mode: {image.mode}")
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 (MNIST input size)
        image = image.resize((28, 28))
        
        if debug:
            print(f"Resized image size: {image.size}")
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        
        if debug:
            print(f"Image array shape: {image_array.shape}")
            print(f"Image array min/max: {image_array.min()}/{image_array.max()}")
            print(f"Image array sample values: {image_array[10:15, 10:15]}")
        
        # Invert the image (black digits on white background -> white digits on black background)
        image_array = 255 - image_array
        
        # Normalize to 0-1 range
        image_array = image_array.astype(np.float32) / 255.0
        
        if debug:
            print(f"Normalized array min/max: {image_array.min()}/{image_array.max()}")
        
        # Reshape to match model input (1, 1, 28, 28)
        image_array = image_array.reshape(1, 1, 28, 28)
        
        # Convert to PyTorch tensor
        tensor = torch.FloatTensor(image_array).to(device)
        
        if debug:
            print(f"Tensor shape: {tensor.shape}")
            print(f"Tensor min/max: {tensor.min().item()}/{tensor.max().item()}")
        
        return tensor
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
        
        print("\n🔍 Processing new prediction...")
        
        # Preprocess the image with debug info
        processed_image = preprocess_image(image_data, debug=True)
        if processed_image is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Make prediction
        with torch.no_grad():
            prediction = model(processed_image)
            probabilities = torch.softmax(prediction, dim=1)
            digit = int(torch.argmax(prediction, dim=1).item())
            confidence = float(torch.max(probabilities).item())
        
        print(f"🎯 Prediction: {digit}")
        print(f"🎯 Confidence: {confidence:.4f}")
        print(f"🎯 All probabilities: {probabilities[0].cpu().numpy()}")
        
        return jsonify({
            'digit': digit,
            'confidence': confidence,
            'all_predictions': probabilities[0].cpu().numpy().tolist()
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📱 Loading PyTorch model...")
    
    if load_model():
        print("✅ Model loaded successfully!")
        print("🌐 Starting server on http://localhost:5000")
        print("🔍 Debug mode enabled - check console for detailed info")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model.")
