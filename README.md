# 🤟 ASL Sign Language Visualizer & Detector

A real-time American Sign Language (ASL) fingerspelling visualizer and translation application powered by **Streamlit**, **MediaPipe Hands**, **OpenCV**, and a **Support Vector Machine (SVM)** classifier.

This project enables users to sign letters using their webcam, translates them in real-time, displays visual landmarks, and provides text-to-speech pronunciation to construct full words and sentences.

---

## 🌟 Features

- **Real-Time Landmark Visualization**: Utilizes MediaPipe's robust hand tracking framework to plot hand nodes and skeletal connections on your camera feed.
- **Topological Feature Extraction**: Converts raw 3D hand coordinates (21 landmarks) into a rotation and translation invariant **121-dimensional feature vector** for accurate classification. Features include:
  - Wrist-relative coordinates normalized for camera distance.
  - Joint angles for major joint triplets (14 features).
  - Tip-to-tip and tip-to-wrist Euclidean distances.
  - Directional ray similarity (cosine similarities of finger bones).
  - Finger crossing logic (e.g., detecting crossed index/middle fingers for 'R').
  - Thumb tuck metrics.
- **Dynamic Text Construction**: 
  - Signs stabilized for over 1.5 seconds are appended to a sentence.
  - Special gestures for `SPACE` and `DELETE` (to remove the last letter or add spacing).
- **Text-to-Speech (TTS)**: Built-in speech feedback using `pyttsx3` to read out recognized letters and completed sentences.
- **Optical Blackout Clearance**: Quickly cover the camera to clear the current sentence and start fresh.
- **Reference UI Guide**: Sidebar/column reference showing the ASL fingerspelling chart for quick learning.

---

## 📁 Project Structure

*   **`app_final.py`**: The primary Streamlit frontend application. Runs the webcam stream, captures frames, extracts hand landmarks, feeds them to the classifier, builds sentences, and reads out words.
*   **`train.py`**: Model training script. Loads raw hand landmarks, applies topological preprocessing, trains an SVM with an RBF kernel, validates accuracy, and dumps the serialized model.
*   **`test_preprocessing.py`**: Diagnostic unit testing suite implementing geometrical assertions (dimensionality checks, intersection logic, bone parallelism).
*   **`asl_landmarks_final.csv`**: CSV dataset containing the hand landmark records for training.
*   **`sign_language_model.pkl`**: Serialized pre-trained SVM model.
*   **`asl_chart.jpg`**: ASL reference chart image.
*   **`.gitignore`**: Excludes system-specific python caches and environment files while retaining project source configurations.

---

## 🚀 Setup & Installation

### Prerequisites
Make sure you have **Python 3.8+** installed. A webcam is required to run the real-time detector.

### 1. Clone the Repository
```bash
git clone https://github.com/darrr1808/Sign-Language-Visualizer.git
cd Sign-Language-Visualizer
```

### 2. Install Dependencies
Install all required libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Use

### Run the App
Launch the Streamlit interface:
```bash
streamlit run app_final.py
```
This will start a local server and automatically open a tab in your web browser (usually at `http://localhost:8501`).

### Train the Model
If you've updated the landmarks dataset (`asl_landmarks_final.csv`) and want to train a fresh model:
```bash
python train.py
```
This trains the SVM classifier, prints the validation accuracy, and exports a new `sign_language_model.pkl`.

### Run Unit Tests
To verify feature extractor logic and landmark geometry:
```bash
python -m unittest test_preprocessing.py
```

---

## 🛠️ Technology Stack

- **Computer Vision / Pose Tracking**: [MediaPipe](https://github.com/google/mediapipe) & [OpenCV](https://opencv.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (SVM Classifier, RBF Kernel)
- **User Interface**: [Streamlit](https://streamlit.io/)
- **Text-to-Speech**: [Pyttsx3](https://pypi.org/project/pyttsx3/)
