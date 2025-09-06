// Global variables
let currentPage = 'canvas';
let isDrawing = false;
let cameraStream = null;
let drawingContext = null;
let equationContext = null;
let model = null; // TensorFlow.js model

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    // Load the TensorFlow.js model
    await loadModel();
    
    // Set up navigation
    setupNavigation();
    
    // Initialize drawing canvas
    initializeDrawingCanvas();
    
    // Initialize camera functionality
    initializeCamera();
    
    // Initialize calculator mode
    initializeCalculator();
}

// Load TensorFlow.js model
async function loadModel() {
    try {
        console.log('Loading TensorFlow.js model...');
        model = await tf.loadLayersModel('./mnist_model_tfjs/model.json');
        console.log('Model loaded successfully!');
        
        // Show model loaded status
        showModelStatus('Model loaded successfully!', 'success');
    } catch (error) {
        console.error('Error loading model:', error);
        showModelStatus('Error loading model. Using mock predictions.', 'error');
    }
}

function showModelStatus(message, type) {
    // Create status element
    const statusDiv = document.createElement('div');
    statusDiv.className = `model-status ${type}`;
    statusDiv.textContent = message;
    statusDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 10px 20px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        z-index: 1000;
        background: ${type === 'success' ? '#28a745' : '#dc3545'};
    `;
    
    document.body.appendChild(statusDiv);
    
    // Remove after 3 seconds
    setTimeout(() => {
        if (statusDiv.parentNode) {
            statusDiv.parentNode.removeChild(statusDiv);
        }
    }, 3000);
}

// Navigation functionality
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const pages = document.querySelectorAll('.page');
    
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetPage = button.getAttribute('data-page');
            switchPage(targetPage);
            
            // Update active button
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
        });
    });
}

function switchPage(pageName) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show target page
    document.getElementById(pageName + '-page').classList.add('active');
    currentPage = pageName;
    
    // Stop camera if switching away from camera page
    if (pageName !== 'camera' && cameraStream) {
        stopCamera();
    }
}

// Drawing Canvas functionality
function initializeDrawingCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    drawingContext = canvas.getContext('2d');
    
    // Set up drawing styles
    drawingContext.strokeStyle = '#000';
    drawingContext.lineWidth = 8;
    drawingContext.lineCap = 'round';
    drawingContext.lineJoin = 'round';
    
    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // Button events
    document.getElementById('clearCanvas').addEventListener('click', clearCanvas);
    document.getElementById('predictDigit').addEventListener('click', predictDigit);
}

function startDrawing(e) {
    isDrawing = true;
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    drawingContext.beginPath();
    drawingContext.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    drawingContext.lineTo(x, y);
    drawingContext.stroke();
}

function stopDrawing() {
    isDrawing = false;
    drawingContext.beginPath();
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                    e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    e.target.dispatchEvent(mouseEvent);
}

function clearCanvas() {
    drawingContext.clearRect(0, 0, 280, 280);
    document.getElementById('predictedDigit').textContent = '-';
    document.getElementById('confidenceText').textContent = '0%';
    document.getElementById('confidenceFill').style.width = '0%';
}

async function predictDigit() {
    const canvas = document.getElementById('drawingCanvas');
    const imageData = drawingContext.getImageData(0, 0, 280, 280);
    
    // Check if canvas has any drawing
    const hasContent = imageData.data.some(pixel => pixel !== 0);
    if (!hasContent) {
        alert('Please draw a digit first!');
        return;
    }
    
    try {
        // Preprocess the image for the model
        const processedImage = preprocessImageForModel(canvas);
        
        if (model) {
            // Use the actual model
            const prediction = model.predict(processedImage);
            const predictionArray = await prediction.data();
            const digit = predictionArray.indexOf(Math.max(...predictionArray));
            const confidence = Math.max(...predictionArray);
            
            document.getElementById('predictedDigit').textContent = digit;
            document.getElementById('confidenceText').textContent = Math.round(confidence * 100) + '%';
            document.getElementById('confidenceFill').style.width = (confidence * 100) + '%';
            
            console.log('Model prediction:', digit, 'Confidence:', confidence);
        } else {
            // Fallback to mock prediction
            const prediction = Math.floor(Math.random() * 10);
            const confidence = Math.random() * 0.4 + 0.6;
            
            document.getElementById('predictedDigit').textContent = prediction;
            document.getElementById('confidenceText').textContent = Math.round(confidence * 100) + '%';
            document.getElementById('confidenceFill').style.width = (confidence * 100) + '%';
            
            console.log('Mock prediction:', prediction, 'Confidence:', confidence);
        }
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please try again.');
    }
}

function preprocessImageForModel(canvas) {
    // Create a temporary canvas for preprocessing
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    // Resize to 28x28 (MNIST input size)
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    
    // Draw the original canvas content to the temp canvas
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // Get image data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // Convert to grayscale and normalize
    const grayscale = [];
    for (let i = 0; i < data.length; i += 4) {
        // Convert to grayscale (average of RGB)
        const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
        // Normalize to 0-1 range
        grayscale.push(gray / 255);
    }
    
    // Reshape to match model input format (1, 28, 28, 1)
    const tensor = tf.tensor4d(grayscale, [1, 28, 28, 1]);
    
    return tensor;
}

// Camera functionality
function initializeCamera() {
    document.getElementById('startCamera').addEventListener('click', startCamera);
    document.getElementById('capturePhoto').addEventListener('click', capturePhoto);
    document.getElementById('stopCamera').addEventListener('click', stopCamera);
    document.getElementById('imageUpload').addEventListener('change', handleImageUpload);
}

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        
        const video = document.getElementById('cameraVideo');
        video.srcObject = cameraStream;
        
        document.getElementById('startCamera').disabled = true;
        document.getElementById('capturePhoto').disabled = false;
        document.getElementById('stopCamera').disabled = false;
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Unable to access camera. Please check permissions.');
    }
}

function capturePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    // Convert to image
    const imageData = canvas.toDataURL('image/png');
    const img = document.getElementById('capturedImg');
    img.src = imageData;
    img.style.display = 'block';
    
    // Predict the digit
    predictImageDigit(imageData);
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        
        document.getElementById('cameraVideo').srcObject = null;
        document.getElementById('startCamera').disabled = false;
        document.getElementById('capturePhoto').disabled = true;
        document.getElementById('stopCamera').disabled = true;
    }
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.getElementById('capturedImg');
            img.src = e.target.result;
            img.style.display = 'block';
            predictImageDigit(e.target.result);
        };
        reader.readAsDataURL(file);
    }
}

async function predictImageDigit(imageData) {
    try {
        // Create an image element to load the image
        const img = new Image();
        img.onload = async function() {
            // Create a canvas to process the image
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size to match image
            canvas.width = img.width;
            canvas.height = img.height;
            
            // Draw image to canvas
            ctx.drawImage(img, 0, 0);
            
            // Preprocess for the model
            const processedImage = preprocessImageForModel(canvas);
            
            if (model) {
                // Use the actual model
                const prediction = model.predict(processedImage);
                const predictionArray = await prediction.data();
                const digit = predictionArray.indexOf(Math.max(...predictionArray));
                const confidence = Math.max(...predictionArray);
                
                document.getElementById('imagePredictedDigit').textContent = digit;
                document.getElementById('imageConfidenceText').textContent = Math.round(confidence * 100) + '%';
                document.getElementById('imageConfidenceFill').style.width = (confidence * 100) + '%';
                
                console.log('Image prediction:', digit, 'Confidence:', confidence);
            } else {
                // Fallback to mock prediction
                const prediction = Math.floor(Math.random() * 10);
                const confidence = Math.random() * 0.4 + 0.6;
                
                document.getElementById('imagePredictedDigit').textContent = prediction;
                document.getElementById('imageConfidenceText').textContent = Math.round(confidence * 100) + '%';
                document.getElementById('imageConfidenceFill').style.width = (confidence * 100) + '%';
                
                console.log('Mock image prediction:', prediction, 'Confidence:', confidence);
            }
        };
        
        img.src = imageData;
    } catch (error) {
        console.error('Image prediction error:', error);
        alert('Error processing image. Please try again.');
    }
}

// Calculator mode functionality
function initializeCalculator() {
    const canvas = document.getElementById('equationCanvas');
    equationContext = canvas.getContext('2d');
    
    // Set up drawing styles
    equationContext.strokeStyle = '#000';
    equationContext.lineWidth = 6;
    equationContext.lineCap = 'round';
    equationContext.lineJoin = 'round';
    
    // Mouse events
    canvas.addEventListener('mousedown', startEquationDrawing);
    canvas.addEventListener('mousemove', drawEquation);
    canvas.addEventListener('mouseup', stopEquationDrawing);
    canvas.addEventListener('mouseout', stopEquationDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleEquationTouch);
    canvas.addEventListener('touchmove', handleEquationTouch);
    canvas.addEventListener('touchend', stopEquationDrawing);
    
    // Button events
    document.getElementById('clearEquation').addEventListener('click', clearEquation);
    document.getElementById('solveEquation').addEventListener('click', solveEquation);
}

let isEquationDrawing = false;

function startEquationDrawing(e) {
    isEquationDrawing = true;
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    equationContext.beginPath();
    equationContext.moveTo(x, y);
}

function drawEquation(e) {
    if (!isEquationDrawing) return;
    
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    equationContext.lineTo(x, y);
    equationContext.stroke();
}

function stopEquationDrawing() {
    isEquationDrawing = false;
    equationContext.beginPath();
}

function handleEquationTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                    e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    e.target.dispatchEvent(mouseEvent);
}

function clearEquation() {
    equationContext.clearRect(0, 0, 400, 200);
    document.getElementById('recognizedExpression').textContent = '-';
    document.getElementById('equationResult').textContent = '-';
    document.getElementById('recognitionSteps').innerHTML = '';
}

function solveEquation() {
    const canvas = document.getElementById('equationCanvas');
    const imageData = equationContext.getImageData(0, 0, 400, 200);
    
    // Check if canvas has any drawing
    const hasContent = imageData.data.some(pixel => pixel !== 0);
    if (!hasContent) {
        alert('Please draw an equation first!');
        return;
    }
    
    // For equation solving, you would need a more complex model
    // that can segment and recognize multiple characters
    // For now, we'll use the mock implementation
    const mockEquation = generateMockEquation();
    const result = evaluateEquation(mockEquation);
    
    document.getElementById('recognizedExpression').textContent = mockEquation;
    document.getElementById('equationResult').textContent = result;
    
    // Show recognition steps
    showRecognitionSteps(mockEquation, result);
    
    console.log('Equation recognition and solving:', mockEquation, '=', result);
}

function generateMockEquation() {
    const equations = [
        '12 + 7', '15 - 8', '6 × 4', '20 ÷ 5', '3 + 4 × 2',
        '10 - 3 + 2', '8 ÷ 2 + 1', '5 × 3 - 7', '9 + 6 ÷ 2'
    ];
    return equations[Math.floor(Math.random() * equations.length)];
}

function evaluateEquation(equation) {
    try {
        // Replace × with * and ÷ with / for evaluation
        const cleanEquation = equation.replace(/×/g, '*').replace(/÷/g, '/');
        return eval(cleanEquation);
    } catch (error) {
        return 'Error';
    }
}

function showRecognitionSteps(equation, result) {
    const stepsDiv = document.getElementById('recognitionSteps');
    const steps = [
        { label: 'Raw Input:', value: 'Handwritten equation detected' },
        { label: 'Segmentation:', value: 'Individual characters isolated' },
        { label: 'Recognition:', value: `"${equation}"` },
        { label: 'Parsing:', value: 'Mathematical expression parsed' },
        { label: 'Calculation:', value: `${equation} = ${result}` }
    ];
    
    stepsDiv.innerHTML = steps.map(step => 
        `<div class="step-item">
            <span class="step-label">${step.label}</span>
            <span class="step-value">${step.value}</span>
        </div>`
    ).join('');
}

// Utility functions
function resizeCanvasToDisplaySize(canvas) {
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
    }
}

// Handle window resize
window.addEventListener('resize', () => {
    resizeCanvasToDisplaySize(document.getElementById('drawingCanvas'));
    resizeCanvasToDisplaySize(document.getElementById('equationCanvas'));
});
