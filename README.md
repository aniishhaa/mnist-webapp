# MNIST Digit Recognition Web Application

A modern web application for recognizing handwritten digits using a PyTorch CNN model and Flask API.

## 🚀 Features

- **Drawing Canvas**: Draw digits directly on a digital canvas
- **Camera Capture**: Take photos of handwritten digits
- **Real-time Prediction**: Get instant predictions with confidence scores
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask (Python)
- **ML Model**: PyTorch CNN
- **Image Processing**: PIL (Python Imaging Library)

## 📋 Prerequisites

- Python 3.8+
- pip (Python package installer)

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aniishhaa/mnist-webapp.git
cd mnist-webapp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Trained Model

Download the `mnist_model.pth` file and place it in the project root directory.

### 4. Run the Flask Server

```bash
python flask_app_debug.py
```

The server will start on `http://localhost:5000`

### 5. Open the Web Application

Open `index.html` in your web browser to start using the application.

## 🎯 How to Use

### Drawing Canvas
1. Click on "Drawing Canvas" tab
2. Draw a digit (0-9) on the white canvas
3. Click "Predict" to get the prediction
4. View the predicted digit and confidence score

### Camera Capture
1. Click on "Camera Capture" tab
2. Click "Start Camera" and allow camera access
3. Draw a digit on paper and hold it up to the camera
4. Click "Capture" to take a photo
5. View the prediction result

## 🏗️ Project Structure

```
mnist-webapp/
├── index.html              # Main web application
├── styles.css              # CSS styling
├── script-with-flask.js    # JavaScript functionality
├── flask_app_debug.py      # Flask API server
├── requirements.txt        # Python dependencies
├── mnist_model.pth         # Trained PyTorch model
├── debug_model.py          # Model testing script
└── README.md              # This file
```

## 🧠 Model Architecture

The CNN model consists of:
- 2 Convolutional layers (16 and 32 filters)
- 2 MaxPooling layers
- 2 Fully connected layers (128 and 10 units)
- ReLU activation functions
- Softmax output layer

## 🔧 Development

### Testing the Model

```bash
python debug_model.py
```

### Running in Debug Mode

The Flask server runs in debug mode by default, showing detailed logs of image processing and predictions.

## 📱 Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**aniishhaa** - [GitHub Profile](https://github.com/aniishhaa)

## 🙏 Acknowledgments

- MNIST dataset for training data
- PyTorch team for the deep learning framework
- Flask team for the web framework
- All contributors and testers

---

⭐ **Star this repository if you found it helpful!**
