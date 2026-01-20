import os
import pickle
import mediapipe as mp
import cv2
from collections import Counter
import random
import numpy as np
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r'C:\Users\Aditi Sharma\Desktop\ISL Project\data'

data = []
labels = []

AUGMENTATION_LIMIT = 500  
AUGMENTATION_METHODS = ['flip', 'rotate', 'brightness']

# Function to augment images
def augment_image(img, method):
    if method == 'flip':
        return cv2.flip(img, 1)
    elif method == 'rotate':
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle=random.randint(-15, 15), scale=1.0)
        return cv2.warpAffine(img, matrix, (w, h))
    elif method == 'brightness':
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

# Extract normalized landmarks for both hands
def extract_hand_landmarks(results):
    data_aux = []
    hand_present = [False, False]

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx > 1:
                break

            x_, y_ = [], []
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
            
            hand_present[idx] = True

    if not hand_present[0]:
        data_aux.extend([0] * 42)
    if not hand_present[1]:
        data_aux.extend([0] * 42)

    return data_aux

for dir_ in tqdm(os.listdir(DATA_DIR), desc="Processing Classes"):
    if os.path.isdir(os.path.join(DATA_DIR, dir_)):
        print(f"Processing data for label: {dir_}")

        class_data = []
        img_paths = [os.path.join(DATA_DIR, dir_, img) for img in os.listdir(os.path.join(DATA_DIR, dir_))]
        
        for img_full_path in img_paths:
            if os.path.isfile(img_full_path):
                img = cv2.imread(img_full_path)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)

                data_aux = extract_hand_landmarks(results)
                if data_aux:
                    class_data.append(data_aux)

        while len(class_data) < AUGMENTATION_LIMIT:
            sample = random.choice(img_paths)
            img = cv2.imread(sample)
            if img is not None:
                method = random.choice(AUGMENTATION_METHODS)
                augmented_img = augment_image(img, method)
                results = hands.process(cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB))

                data_aux = extract_hand_landmarks(results)
                if data_aux:
                    class_data.append(data_aux)

        data.extend(class_data)
        labels.extend([dir_] * len(class_data))

label_counts = Counter(labels)
print("Label distribution:", label_counts)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Balanced data saved to 'data.pickle'.")
