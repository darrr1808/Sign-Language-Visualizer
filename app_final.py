import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import joblib
import time
import pyttsx3
import threading
import os

@st.cache_resource
def load_models():
    return joblib.load('sign_language_model.pkl')

model = load_models()

def extract_features(coords):
    """121 Dimensional Extraction"""
    wrist = coords[0]
    relative = coords - wrist
    max_val = np.max(np.abs(relative))
    normalized = relative / max_val if max_val > 0 else relative
    
    thumb_tip = normalized[4]
    thumb_distances = np.linalg.norm(normalized - thumb_tip, axis=1)
    
    angles = []
    joint_triplets = [(1,2,3), (2,3,4), (5,6,7), (6,7,8), (9,10,11), (10,11,12), (13,14,15), (14,15,16), (17,18,19), (18,19,20), (1,0,5), (5,0,9), (9,0,13), (13,0,17)]
    for a, b, c in joint_triplets:
        v1, v2 = normalized[a] - normalized[b], normalized[c] - normalized[b]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0: angles.append(0.0)
        else: angles.append(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))
        
    tips = [4, 8, 12, 16, 20]
    tip_distances = [np.linalg.norm(normalized[tips[i]] - normalized[tips[j]]) for i in range(len(tips)) for j in range(i + 1, len(tips))]
    wrist_distances = [np.linalg.norm(normalized[tip] - normalized[0]) for tip in tips]
    
    rays = [normalized[8]-normalized[5], normalized[12]-normalized[9], normalized[16]-normalized[13], normalized[20]-normalized[17]]
    ray_sims = []
    for i in range(len(rays) - 1):
        n1, n2 = np.linalg.norm(rays[i]), np.linalg.norm(rays[i+1])
        ray_sims.append(np.dot(rays[i], rays[i+1]) / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0)
        
    mcp_x_diff = normalized[5][0] - normalized[9][0]
    tip_x_diff = normalized[8][0] - normalized[12][0]
    is_crossed = 1.0 if (mcp_x_diff * tip_x_diff < 0) else 0.0
    
    thumb_tucks = [
        np.linalg.norm(normalized[4] - normalized[6]), 
        np.linalg.norm(normalized[4] - normalized[10]),
        np.linalg.norm(normalized[4] - normalized[14]),
        np.linalg.norm(normalized[4] - normalized[8])  
    ]
        
    return np.concatenate((
        normalized.flatten(), thumb_distances, np.array(angles), 
        np.array(tip_distances), np.array(wrist_distances), 
        np.array(ray_sims), np.array([is_crossed]), np.array(thumb_tucks)
    )).reshape(1, -1)

def is_optical_blackout(frame_matrix, threshold=25):
    return np.mean(frame_matrix) < threshold

def speak_word(text):
    def tts_task():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            pass
    threading.Thread(target=tts_task, daemon=True).start()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

st.set_page_config(page_title="ASL Tutor", layout="wide")
st.title("ASL Sign Language Detection")
col1, col2 = st.columns([2, 1])

with col1: frame_placeholder = st.empty()
with col2:
    st.markdown("### Reference")
    if os.path.exists("asl_chart.jpg"): st.image("asl_chart.jpg", use_container_width=True)
    st.markdown("### Detected Metrics")
    letter_display = st.empty()
    word_display = st.empty()

cap = cv2.VideoCapture(0)
constructed_word, current_stable_letter, predicted_letter = "", "-", "-"
last_detected_time = time.time()
flash_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    if is_optical_blackout(frame, threshold=25):
        constructed_word, current_stable_letter, predicted_letter = "", "-", "-"
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "CLEARED", (int(w/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        coords_3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        features = extract_features(coords_3d)
        predicted_letter = model.predict(features)[0]
        
        if predicted_letter != current_stable_letter:
            current_stable_letter = predicted_letter
            last_detected_time = time.time()
        elif time.time() - last_detected_time > 1.5:  
            flash_counter = 5 
            if predicted_letter in ['DELETE', 'DEL']: constructed_word = constructed_word[:-1]
            elif predicted_letter == 'SPACE': 
                speak_word(constructed_word)
                constructed_word += " "
            else: constructed_word += predicted_letter
            last_detected_time = time.time() 
            speak_word(predicted_letter)
    else: predicted_letter = "-"
        
    if flash_counter > 0:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
        flash_counter -= 1

    cv2.putText(frame, f"Detecting: {predicted_letter}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Text: {constructed_word}", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    letter_display.markdown(f"**Current Sign:** {predicted_letter}")
    word_display.markdown(f"**Sentence:** {constructed_word}")

cap.release()