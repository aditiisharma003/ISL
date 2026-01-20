import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import os

app = Flask(__name__)

# Load the trained model with error handling
model = None
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("⚠️ Warning: model.p not found. Predictions will not work.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Define labels (A-Z and 0-9)
labels_dict = {str(i): str(i) for i in range(10)} 
labels_dict.update({chr(65 + i): chr(65 + i) for i in range(26)})  

# Define gestures that require both hands
two_handed_gestures = {'A','B','D','E','F','G','H','J','K','M','N','P','Q','R','S','T','W','X','Y','Z'}  

def process_frame(image_data):
    """Process a single frame and return prediction"""
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        data_aux = []
        hand_count = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_count += 1
                x_ = []
                y_ = []
                
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                
                # Normalize coordinates
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))
        
        # Ensure 84 features
        if len(data_aux) == 42:
            data_aux.extend([0] * 42)
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]
        
        # Make prediction
        if model and len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(prediction[0], "?")
            
            # Check for two-handed gestures
            if predicted_character in two_handed_gestures and hand_count < 2:
                predicted_character = "?"
        else:
            predicted_character = "?" if not model else "No hands"
        
        return {
            "prediction": predicted_character,
            "hands_detected": hand_count
        }
    
    except Exception as e:
        print(f"Processing error: {e}")
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction from base64 image"""
    data = request.get_json()
    
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    result = process_frame(data['image'])
    return jsonify(result)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
