import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

def extract_features(coords):
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
        np.linalg.norm(normalized[4] - normalized[6]),  # T
        np.linalg.norm(normalized[4] - normalized[10]), # N
        np.linalg.norm(normalized[4] - normalized[14]), # M
        np.linalg.norm(normalized[4] - normalized[8])   # D
    ]
        
    return np.concatenate((
        normalized.flatten(), 
        thumb_distances,
        np.array(angles), 
        np.array(tip_distances), 
        np.array(wrist_distances),
        np.array(ray_sims), 
        np.array([is_crossed]), 
        np.array(thumb_tucks) 
    )) 

print("Loading dataset...")
df = pd.read_csv('asl_landmarks_final.csv')

print("Purging dynamic signs (J, Z)...")
df = df[~df['label'].isin(['J', 'Z', 'j', 'z'])]

X_raw = df.drop('label', axis=1).values
y = df['label'].values

num_samples = X_raw.shape[0]
X_3d = X_raw.reshape(num_samples, 21, 3)

print("Applying 121-Dimension Topological Extraction...")
X_processed = np.array([extract_features(sample) for sample in X_3d])

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, stratify=y, random_state=42
)

model = SVC(kernel='rbf', C=50, gamma='scale', probability=True)
model.fit(X_train, y_train)

print(f"Validation Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
joblib.dump(model, 'sign_language_model.pkl')
print("Model saved.")