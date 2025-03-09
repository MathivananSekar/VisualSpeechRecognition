import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse

from src.model.resnet_ctc import LipReadingModel
from src.training.vocab import VOCAB, vocab_size
from src.preprocessing.lip_detection import detect_lips
from src.preprocessing.extract_frames import extract_frames


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_video(frames, resize=(32, 64)):
    """
    Preprocess frames into consistent shape (grayscale -> 3 channels).
    frames: list of 2D grayscale images
    Desired final shape: (1, 3, num_frames, 32, 64)
    """
    # 1) Resize each frame to (32, 64) => (height=32, width=64)
    processed_list = []
    for frame in frames:
        # cv2.resize expects (width, height)
        frame_resized = cv2.resize(frame, (resize[1], resize[0]))
        processed_list.append(frame_resized)

    # 2) Convert to NumPy array => shape: (num_frames, 32, 64)
    processed = np.array(processed_list, dtype=np.uint8)

    # 3) Convert to Float32 tensor => (num_frames, 32, 64)
    processed = torch.tensor(processed, dtype=torch.float32)

    # 4) Add a batch dimension => (1, num_frames, 32, 64)
    processed = processed.unsqueeze(0)

    # 5) Add a channel dimension => (1, 1, num_frames, 32, 64)
    processed = processed.unsqueeze(1)

    # 6) Repeat the single channel 3 times => (1, 3, num_frames, 32, 64)
    processed = processed.repeat(1, 3, 1, 1, 1)

    return processed

def greedy_decode(ctc_probs, blank_index=0):
    """
    ctc_probs: (time, vocab_size) after softmax 
    Perform simple greedy decoding:
    - pick argmax at each time step
    - skip repeated tokens and blank tokens
    """
    argmax_seq = torch.argmax(ctc_probs, dim=-1).cpu().numpy()

    decoded = []
    prev = None
    for idx in argmax_seq:
        if idx != blank_index and idx != prev:
            decoded.append(idx)
        prev = idx

    return decoded

def labels_to_text(indices):
    """
    Convert numeric label indices back to words/phonemes 
    from the VOCAB dictionary (inverted).
    """
    idx_to_token = {v: k for k, v in VOCAB.items()}

    words = []
    for i in indices:
        token = idx_to_token.get(i, "<unk>")
        # Filter out blank/sil if you like
        if token not in ["<blank>", "sil"]:
            words.append(token)
    return words

def predict_lip_reading(video_path, checkpoint_path):
    """
    1) Load model
    2) Extract/preprocess frames
    3) Forward pass -> ctc logits
    4) Greedy decode
    5) Convert to text
    """
    # 1) Load model
    model = LipReadingModel(vocab_size())
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2) Extract & preprocess
    raw_frames = extract_frames(video_path)   # grayscale frames
    lip_frames = detect_lips(raw_frames)      # if you do real lip cropping
    input_tensor = preprocess_video(lip_frames)  # (1, 3, T, 32, 64)
    input_tensor = input_tensor.to(DEVICE)

    # 3) Forward pass -> (time, batch=1, vocab_size)
    with torch.no_grad():
        logits = model(input_tensor)  # shape: (T, 1, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)  # (T, 1, vocab_size)
    
    # 4) Greedy decode
    log_probs = log_probs.squeeze(1)     # => (T, vocab_size)
    decoded_indices = greedy_decode(log_probs, blank_index=0)

    # 5) Convert to text
    words = labels_to_text(decoded_indices)
    predicted_text = " ".join(words)
    return predicted_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict video using saved model.")
    parser.add_argument("--video_path", type=str, required=True, help="Video path for prediction.")
    args = parser.parse_args()
    print(f"Prediction for video: {args.video_path}")

    checkpoint = "checkpoints/epoch_50.pth"  # Or whichever checkpoint you want to load
    prediction = predict_lip_reading(args.video_path, checkpoint)
    print("Predicted:", prediction)
