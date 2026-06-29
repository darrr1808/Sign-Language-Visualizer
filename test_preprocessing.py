import numpy as np
import unittest

def extract_features(coords):
    """validation."""
    wrist = coords[0]
    relative = coords - wrist
    max_val = np.max(np.abs(relative))
    normalized = relative / max_val if max_val > 0 else relative
    
    thumb_tip = normalized[4]
    thumb_distances = np.linalg.norm(normalized - thumb_tip, axis=1)
    
    angles = []
    joint_triplets = [(1,2,3), (2,3,4), (5,6,7), (6,7,8), (9,10,11), (10,11,12), (13,14,15), (14,15,16), (17,18,19), (18,19,20), (1,0,5), (5,0,9), (9,0,13), (13,0,17)]
    for a, b, c in joint_triplets:
        v1 = normalized[a] - normalized[b]
        v2 = normalized[c] - normalized[b]
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            angles.append(0.0)
            continue
        dot_prod = np.dot(v1, v2)
        cos_angle = np.clip(dot_prod / (v1_norm * v2_norm), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_angle)))
        
    tips = [4, 8, 12, 16, 20]
    tip_distances = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            tip_distances.append(np.linalg.norm(normalized[tips[i]] - normalized[tips[j]]))
            
    wrist_distances = [np.linalg.norm(normalized[tip] - normalized[0]) for tip in tips]
        
    return np.concatenate((normalized.flatten(), thumb_distances, np.array(angles), np.array(tip_distances), np.array(wrist_distances))).reshape(1, -1)


class TestInferencePipeline(unittest.TestCase):
    
    def test_fingertip_matrix_dimensions(self):
        """113 metrics."""
        np.random.seed(99)
        coords = np.random.rand(21, 3)
        features = extract_features(coords)
        
        self.assertEqual(features.shape[1], 113, "Feature vector failed dimension expansion.")

    def test_finger_intersection_geometry(self):
        """Proof: Simulating the 'R' sign (crossed index and middle fingers) yields near-zero tip distance."""
        coords = np.zeros((21, 3))
        coords[8] = [0.5, 0.5, 0.5]
        coords[12] = [0.5, 0.5, 0.5]
        
        coords[0] = [0.0, 0.0, 0.0]
        coords[1] = [1.0, 1.0, 1.0] 
        
        features = extract_features(coords)
        
        index_middle_distance = features[0][102]
        
        self.assertAlmostEqual(index_middle_distance, 0.0, msg="Intersection failed to output 0 distance.")

    def test_finger_ray_parallelism(self):
        """Proof: U vs V. Parallel fingers yield a cosine similarity near 1.0."""
        coords = np.zeros((21, 3))
        coords[5] = [0.0, 0.0, 0.0]
        coords[8] = [0.0, 1.0, 0.0]
        coords[9] = [1.0, 0.0, 0.0]
        coords[12] = [1.0, 1.0, 0.0]
        coords[0] = [-1.0, -1.0, -1.0]
        
        features = extract_features(coords)
        
        index_middle_parallelism = features[0][113]
        
        self.assertAlmostEqual(index_middle_parallelism, 1.0, 
            msg="Ray Directionality failed. Parallel fingers did not return 1.0.")

    def test_intersection_logic(self):
        """Proof: Cross-Detection. Simulating the 'R' sign correctly triggers the intersection boolean."""
        coords = np.zeros((21, 3))
        coords[5] = [0.0, 0.0, 0.0] 
        coords[9] = [1.0, 0.0, 0.0]
        
        coords[8] = [1.0, 1.0, 0.0] 
        coords[12] = [0.0, 1.0, 0.0]
        
        coords[0] = [-1.0, -1.0, -1.0]
        
        features = extract_features(coords)
        
        is_crossed = features[0][116]
        
        self.assertEqual(is_crossed, 1.0, 
            msg="fingers were not detected.")

if __name__ == '__main__':
    unittest.main()