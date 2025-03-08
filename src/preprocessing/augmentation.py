import numpy as np
import cv2

def augment_frames(frames):
    """Apply data augmentation to lip frames."""
    augmented_frames = []
    
    for frame in frames:
        # Horizontal Flip (50% probability)
        if np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)

        # Color Jitter (Brightness adjustment)
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frame = np.clip(frame * factor, 0, 255).astype(np.uint8)

        augmented_frames.append(frame)

    return np.array(augmented_frames)
