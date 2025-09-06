# MNIST Model Integration Guide

## Option 1: TensorFlow.js Integration (Recommended)

### Step 1: Convert Your Model in Colab

Add this code to your Colab notebook after training your model:

```python
import tensorflow as tf
import tensorflowjs as tfjs

# Assuming your model is saved as 'model'
# Convert to TensorFlow.js format
tfjs.converters.save_keras_model(model, './mnist_model_tfjs')

# Also save as SavedModel for backup
model.save('./mnist_model_savedmodel')
```

### Step 2: Download the Model Files

1. Download the entire `mnist_model_tfjs` folder from Colab
2. Place it in your web app directory
3. The folder should contain:
   - `model.json`
   - `weights.bin` (or multiple .bin files)

### Step 3: Update the Web App

The web app will be updated to load and use your TensorFlow.js model.

## Option 2: Flask API Backend

### Step 1: Create Flask API

Create a new file `app.py`:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Load your model
model = tf.keras.models.load_model('path_to_your_model.h5')

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    try:
        data = request.json
        image_data = data['image']
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image (resize to 28x28, convert to grayscale, normalize)
        image = image.convert('L').resize((28, 28))
        image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
        
        # Make prediction
        prediction = model.predict(image_array)
        digit = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        return jsonify({
            'digit': int(digit),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Step 2: Install Dependencies

```bash
pip install flask flask-cors tensorflow pillow numpy
```

### Step 3: Run the API

```bash
python app.py
```

## Option 3: Cloud Deployment

### Google Cloud AI Platform
1. Upload your model to Google Cloud Storage
2. Deploy using AI Platform
3. Use the REST API endpoint

### Hugging Face Spaces
1. Create a new Space
2. Upload your model files
3. Create a Gradio or Streamlit interface
4. Use the API endpoint

## Which Option Should You Choose?

- **TensorFlow.js**: Best for simple models, runs in browser, no server needed
- **Flask API**: Good for complex models, full control, runs locally
- **Cloud Deployment**: Best for production, scalable, but requires cloud setup

## Next Steps

1. Choose your preferred option
2. Follow the specific steps for that option
3. Update the web app JavaScript to use your model
4. Test the integration

Let me know which option you'd like to proceed with!
