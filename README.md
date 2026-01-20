# Sign Language Recognition System

A real-time sign language recognition system using MediaPipe and Machine Learning.

## Features
- Real-time hand gesture recognition
- Support for letters (A-Z) and numbers (0-9)
- Two-handed gesture detection
- Web-based interface using Flask

## Requirements
```
flask
opencv-python
mediapipe
numpy
scikit-learn
pickle
tqdm
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

## Usage
- Navigate to `http://localhost:5000` in your browser
- Allow camera access
- Show hand gestures to the camera for recognition

## Project Structure
- `app.py` - Main Flask application
- `create_dataset.py` - Dataset creation and augmentation
- `model.p` - Trained model (not included)
- `data.pickle` - Training data (not included)