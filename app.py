import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define labels (A-Z and 0-9)
labels_dict = {str(i): str(i) for i in range(10)} 
labels_dict.update({chr(65 + i): chr(65 + i) for i in range(26)})  

# Define gestures that require both hands
two_handed_gestures = {'A','B','D','E','F','G','H','J','K','M','N','P','Q','R','S','T','W','X','Y','Z'}  

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        data_aux = []  
        combined_rect = [W, H, 0, 0]  
        hand_count = 0  

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_count += 1
                x_ = []
                y_ = []

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                # Normalize coordinates
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                # Update combined bounding box
                min_x = int(min(x_) * W)
                min_y = int(min(y_) * H)
                max_x = int(max(x_) * W)
                max_y = int(max(y_) * H)
                combined_rect[0] = min(combined_rect[0], min_x)
                combined_rect[1] = min(combined_rect[1], min_y)
                combined_rect[2] = max(combined_rect[2], max_x)
                combined_rect[3] = max(combined_rect[3], max_y)

        # Ensure the input has 84 landmarks (42 for each hand)
        if len(data_aux) == 42:  
            data_aux.extend([0] * 42)
        elif len(data_aux) > 84:  
            data_aux = data_aux[:84]

        # Make predictions using the model
        try:
            if len(data_aux) == 84: 
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(prediction[0], "?")

                # Check for two-handed gestures
                if predicted_character in two_handed_gestures and hand_count < 2:
                    predicted_character = "?" 

            else:
                predicted_character = "?"
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_character = "?"

        # Draw the combined bounding box and prediction
        if combined_rect[2] > combined_rect[0] and combined_rect[3] > combined_rect[1]:
            cv2.rectangle(frame, (combined_rect[0], combined_rect[1]),
                          (combined_rect[2], combined_rect[3]), (0, 255, 0), 3)
            cv2.putText(frame, predicted_character, (combined_rect[0], combined_rect[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)