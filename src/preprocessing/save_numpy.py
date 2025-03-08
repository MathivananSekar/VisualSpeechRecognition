import numpy as np

def save_numpy(lip_frames, labels, output_path):
    """Save processed video frames and labels as NumPy."""
    np.save(output_path, {"frames": lip_frames, "labels": labels})
    print(f"Saved: {output_path}")
