import dlib
import numpy as np
import cv2

# Load Dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/face_landmarks/shape_predictor_68_face_landmarks.dat")  # Download this model

LIP_INDICES = list(range(48, 68))  # Lip region indices

def detect_lips(frames):
    """Detect and crop lips from frames using Dlib landmarks."""
    lip_frames = []
    
    for frame in frames:
        faces = detector(frame)
        if len(faces) == 0:
            continue  # Skip if no face detected
        
        shape = predictor(frame, faces[0])  # Detect landmarks
        lip_points = np.array([[shape.part(i).x, shape.part(i).y] for i in LIP_INDICES])
        
        x_min, y_min = np.min(lip_points, axis=0)
        x_max, y_max = np.max(lip_points, axis=0)
        
        lip_roi = frame[y_min:y_max, x_min:x_max]  # Crop lip region
        lip_frames.append(cv2.resize(lip_roi, (64, 32)))  # Resize lips to standard size
    
    return np.array(lip_frames)  # Convert to NumPy array
