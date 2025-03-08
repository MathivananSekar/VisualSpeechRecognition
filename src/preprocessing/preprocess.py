import os
import numpy as np
import argparse
from glob import glob
from src.preprocessing.extract_frames import extract_frames
from src.preprocessing.lip_detection import detect_lips
from src.preprocessing.align_parser import parse_align_file
from src.preprocessing.augmentation import augment_frames
from src.preprocessing.save_numpy import save_numpy

DATA_DIR = "data/raw"  # Raw dataset path
OUTPUT_DIR = "data/processed"  # Processed dataset path

def process_video(video_path, align_path, output_path):
    """Process a single video: extract frames, crop lips, align text, and save NumPy file."""
    print(f"Processing video: {video_path}")

    frames = extract_frames(video_path)
    lip_frames = detect_lips(frames)
    lip_frames = augment_frames(lip_frames)
    labels = parse_align_file(align_path)
    print(f"Extracted actual frames - {len(frames)} ,lip frames - {len(lip_frames)}, labels {len(labels)} from {video_path}.")

    save_numpy(lip_frames, labels, output_path)

def process_speaker(speaker):
    """Process all videos of a given speaker."""
    speaker_dir = os.path.join(DATA_DIR, speaker)
    video_files = glob(os.path.join(speaker_dir, "videos", "*.mpg"))

    for video_path in video_files:
        video_name = os.path.basename(video_path).replace(".mpg", "")
        align_path = os.path.join(speaker_dir, "alignments", f"{video_name}.align")
        
        if not os.path.exists(align_path):
            print(f"Skipping {video_name}, alignment file missing.")
            continue

        output_path = os.path.join(OUTPUT_DIR, speaker, "npy", f"{video_name}.npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        process_video(video_path, align_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess videos using mouth detection/cropping.")
    parser.add_argument("--spk_id", type=str, required=True, help="Speaker ID, e.g., s1.")
    args = parser.parse_args()
    print(f"Processing speaker: {args.spk_id}")
    process_speaker(args.spk_id)
