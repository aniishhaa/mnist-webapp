# MNIST Digit Recognition Web Application

A modern, full-stack web application for recognizing handwritten digits using a custom PyTorch CNN model and Flask API. This project demonstrates the complete pipeline from model training in Google Colab to deployment in a user-friendly web interface.

## 🚀 Features

- **🎨 Drawing Canvas**: Draw digits directly on a digital canvas with smooth, responsive drawing
- **📷 Camera Capture**: Take photos of handwritten digits using your device's camera
- **⚡ Real-time Prediction**: Get instant predictions with confidence scores
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **🔧 Debug Mode**: Detailed logging for development and troubleshooting
- **🎯 High Accuracy**: Custom-trained CNN model with 99%+ accuracy on MNIST data

## 🛠️ Technology Stack

### Frontend
- **HTML5**: Semantic markup and canvas API
- **CSS3**: Modern styling with gradients and animations
- **JavaScript (ES6+)**: Interactive functionality and API communication

### Backend
- **Flask**: Lightweight Python web framework
- **PyTorch**: Deep learning framework for model inference
- **PIL (Pillow)**: Image processing and manipulation
- **NumPy**: Numerical computations

### Machine Learning
- **Custom CNN Architecture**: 2-layer convolutional neural network
- **MNIST Dataset**: 60,000 training images, 10,000 test images
- **Mini-batch Gradient Descent**: Efficient training with Adam optimizer

## 📊 Model Performance

- **Training Accuracy**: 99.32%
- **Test Accuracy**: 99.13%
- **Model Size**: ~2.5MB
- **Inference Time**: <50ms per prediction
- **Confidence Scores**: 60-100% for well-drawn digits

## 🧠 Model Architecture

```python
CNN(
  Conv2d(1, 16, kernel_size=5, padding=2)    # 28x28x1 → 28x28x16
  MaxPool2d(2, 2)                            # 28x28x16 → 14x14x16
  Conv2d(16, 32, kernel_size=5, padding=2)   # 14x14x16 → 14x14x32
  MaxPool2d(2, 2)                            # 14x14x32 → 7x7x32
  Flatten()                                  # 7x7x32 → 1568
  Linear(1568, 128)                          # 1568 → 128
  Linear(128, 10)                            # 128 → 10 (digits 0-9)
)
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package installer
- **Web Browser**: Chrome, Firefox, Safari, or Edge
- **Camera**: For camera capture feature (optional)

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aniishhaa/HDR.git
cd HDR
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Trained Model

The trained model (`mnist_model.pth`) should be included in the repository. If not, you can train your own model using the Colab notebook.

### 4. Run the Flask Server

```bash
python flask_app_debug.py
```

The server will start on `http://localhost:5000`

### 5. Open the Web Application

Open `index.html` in your web browser to start using the application.

## 🎯 How to Use

### Drawing Canvas
1. Navigate to the "Drawing Canvas" tab
2. Draw a digit (0-9) on the white canvas using your mouse, stylus, or finger
3. Click "Predict" to get the prediction
4. View the predicted digit and confidence score
5. Use "Clear" to start over

### Camera Capture
1. Navigate to the "Camera Capture" tab
2. Click "Start Camera" and allow camera access when prompted
3. Draw a digit on paper and hold it up to the camera
4. Click "Capture" to take a photo
5. View the prediction result
6. Use "Upload Image" to test with existing photos

## 🏗️ Project Structure

```
HDR/
├── index.html                  # Main web application
├── styles.css                  # CSS styling and responsive design
├── script-with-flask.js        # JavaScript functionality and API calls
├── flask_app_debug.py          # Flask API server with debug logging
├── requirements.txt            # Python dependencies
├── mnist_model.pth             # Trained PyTorch model weights
├── debug_model.py              # Model testing and validation script
├── model-integration-guide.md  # Integration documentation
└── README.md                   # This file
```

## 🔬 Model Training

The model was trained using Google Colab with the following approach:

### Training Process
1. **Data Loading**: MNIST dataset with 60,000 training samples
2. **Data Preprocessing**: Normalization and augmentation
3. **Model Architecture**: Custom CNN with 2 convolutional layers
4. **Training**: 5 epochs with Adam optimizer (lr=0.001)
5. **Validation**: 10,000 test samples for accuracy measurement
6. **Model Saving**: State dict saved as `mnist_model.pth`

### Colab Notebook
📓 **Training Notebook**: [View on Google Colab](https://colab.research.google.com/drive/19XcoZCNaiVAn0nJ--0izpGC0s1AewB7W?usp=sharing)

The notebook includes:
- Complete training pipeline
- Model architecture definition
- Training visualization
- Performance metrics
- Model conversion utilities

## 🔧 Development

### Testing the Model

```bash
python debug_model.py
```

This script tests the model with various inputs and shows:
- Random input predictions
- Zero input behavior
- Center dot test
- All probability distributions

### Debug Mode

The Flask server runs in debug mode by default, showing:
- Image preprocessing details
- Model input/output shapes
- Prediction confidence scores
- Error handling and logging

### API Endpoints

- `GET /health` - Server health check
- `POST /predict_digit` - Predict digit from image data
- `POST /predict_equation` - (Future: equation recognition)

## 📱 Browser Compatibility

- ✅ **Chrome** (recommended)
- ✅ **Firefox**
- ✅ **Safari**
- ✅ **Edge**
- ✅ **Mobile browsers** (iOS Safari, Chrome Mobile)

## 🚀 Future Enhancements

### Phase 1: Hand Movement Recognition
- **Real-time hand tracking** using computer vision
- **Gesture recognition** for drawing in air
- **3D space digit recognition** from hand movements
- **Integration with webcam** for continuous monitoring

### Phase 2: Mathematical Operations
- **Symbol recognition** for +, -, ×, ÷, = operators
- **Multi-digit number** recognition and segmentation
- **Equation parsing** and mathematical evaluation
- **Step-by-step solution** display

### Phase 3: English Handwriting
- **Character recognition** for A-Z, a-z letters
- **Word recognition** and spell checking
- **Sentence parsing** and grammar analysis
- **Handwriting style** adaptation and learning

### Phase 4: Advanced Features
- **Multi-language support** for different scripts
- **Custom model training** interface
- **Batch processing** for multiple images
- **Cloud deployment** with scalable infrastructure

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution
- 🎨 **UI/UX improvements**
- 🧠 **Model architecture enhancements**
- 📱 **Mobile optimization**
- 🔧 **Performance optimizations**
- 📚 **Documentation updates**
- 🐛 **Bug fixes and testing**

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**aniishhaa** - [GitHub Profile](https://github.com/aniishhaa)

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun and the MNIST team for the foundational dataset
- **PyTorch Team**: For the excellent deep learning framework
- **Flask Team**: For the lightweight and flexible web framework
- **Google Colab**: For providing free GPU access for model training
- **Open Source Community**: For the tools and libraries that made this possible

## 📈 Project Statistics

- **Lines of Code**: 500+ (Python + JavaScript + HTML/CSS)
- **Model Parameters**: ~50,000
- **Training Time**: ~5 minutes on Colab GPU
- **Inference Speed**: <50ms per prediction
- **Accuracy**: 99%+ on MNIST test set

---

⭐ **Star this repository if you found it helpful!**

🔗 **Connect with me**: [GitHub](https://github.com/aniishhaa) | [LinkedIn](https://linkedin.com/in/aniishhaa)

---

*This project demonstrates the complete pipeline from data preprocessing and model training to web deployment, showcasing modern machine learning and web development practices.*
