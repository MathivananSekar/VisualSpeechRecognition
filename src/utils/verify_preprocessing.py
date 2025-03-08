import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# Path to processed data
PROCESSED_DIR = "data/processed/s1/npy"  # Change 's1' if needed

def load_random_sample():
    """Load a random processed NumPy file and display its content."""
    npy_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".npy")]
    if not npy_files:
        print("No processed files found!")
        return

    # Choose a random file
    sample_file = random.choice(npy_files)
    sample_path = os.path.join(PROCESSED_DIR, sample_file)

    # Load data
    data = np.load(sample_path, allow_pickle=True).item()
    frames, labels = data["frames"], data["labels"]

    print(f"Loaded: {sample_file}")
    print(f"Frames Shape: {frames.shape}")  # (num_frames, height, width)
    print(f"Labels: {labels}")

    return frames, labels, sample_file

def show_frames(frames, sample_file, num_samples=25):
    """Display a few random lip frames from the processed file."""
    num_frames = frames.shape[0]
    indices = np.linspace(0, num_frames-1, num_samples, dtype=int)  # Evenly spaced frames

    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(frames[idx], cmap="gray")
        plt.axis("off")
        plt.title(f"Frame {idx}")

    plt.suptitle(f"Sample: {sample_file}", fontsize=14)
    plt.show()

if __name__ == "__main__":
    frames, labels, sample_file = load_random_sample()
    if frames is not None:
        show_frames(frames, sample_file)
