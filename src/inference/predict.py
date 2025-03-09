import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import kenlm

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

def kenlm_score(kenlm_model, sentence):
    """
    Get LM score for a sentence from KenLM.
    KenLM returns log10 probability; convert to natural log.
    """
    score = kenlm_model.score(sentence, bos=True, eos=True)
    return score * np.log(10)

def ctc_shallow_fusion_beam_search(ctc_log_probs, kenlm_model, idx_to_token, blank_idx=0, beam_size=3, alpha=0.5):
    """
    Beam search decoding that integrates KenLM scores.
    
    Args:
        ctc_log_probs: Tensor of shape (T, vocab_size) containing CTC log probabilities (natural log).
        kenlm_model: Loaded KenLM model.
        idx_to_token: Dictionary mapping token index -> word.
        blank_idx: Index of blank token.
        beam_size: Number of beams to keep.
        alpha: Weight of the LM score.
    
    Returns:
        List of tokens (indices) corresponding to the best hypothesis.
    """
    T, vocab_size = ctc_log_probs.shape
    # Each beam: dict with "hyp": list of token indices, "score": cumulative score, "sentence": current sentence string.
    beams = [{"hyp": [], "score": 0.0, "sentence": ""}]
    
    for t in range(T):
        new_beams = []
        for beam in beams:
            for token in range(vocab_size):
                ctc_score = ctc_log_probs[t, token].item()
                new_hyp = beam["hyp"] + [token]
                word = idx_to_token.get(token, "<unk>")
                # For blank tokens, do not update sentence.
                if token == blank_idx:
                    new_sentence = beam["sentence"]
                else:
                    # Append a space and the new word.
                    new_sentence = (beam["sentence"] + " " + word).strip()
                lm_score = kenlm_score(kenlm_model, new_sentence) if new_sentence != "" else 0.0
                combined_score = beam["score"] + ctc_score + alpha * lm_score
                new_beams.append({
                    "hyp": new_hyp,
                    "score": combined_score,
                    "sentence": new_sentence
                })
        # Keep top beam_size beams.
        new_beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)
        beams = new_beams[:beam_size]
    
    best_beam = beams[0]
    return best_beam["hyp"]

def predict_lip_reading(video_path, checkpoint_path,kenlm_model_path=None, beam_size=3, alpha=0.5):
    """
    1) Load lip reading model.
    2) Preprocess video and get CTC log probabilities.
    3) Decode using greedy decoding.
    4) Decode using beam search with KenLM shallow fusion (if kenlm_model_path provided).
    Returns both predictions.
    """
    # 1) Load model
    model = LipReadingModel(vocab_size())
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2) Extract and preprocess frames
    raw_frames = extract_frames(video_path)
    lip_frames = detect_lips(raw_frames)
    input_tensor = preprocess_video(lip_frames)
    input_tensor = input_tensor.to(DEVICE)

    # 3) Forward pass -> (T, 1, vocab_size)
    with torch.no_grad():
        logits = model(input_tensor)
        log_probs = F.log_softmax(logits, dim=-1)  # (T, 1, vocab_size)
    log_probs = log_probs.squeeze(1)  # (T, vocab_size)

    # Greedy decoding (before LM fusion)
    greedy_indices = greedy_decode(log_probs, blank_index=0)
    greedy_text = labels_to_text(greedy_indices)

    final_text = greedy_text  # default final is greedy
    # If kenlm_model_path is provided, perform shallow fusion beam search
    if kenlm_model_path is not None:
        kenlm_model = kenlm.Model(kenlm_model_path)
        idx_to_token = {v: k for k, v in VOCAB.items()}
        beam_indices = ctc_shallow_fusion_beam_search(log_probs, kenlm_model, idx_to_token, blank_idx=0, beam_size=beam_size, alpha=alpha)
        final_text = labels_to_text(beam_indices)

    return greedy_text, final_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict video using lip reading model with LM fusion")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_50.pth", help="Path to lip reading model checkpoint")
    parser.add_argument("--kenlm_model", type=str, default=None, help="Path to KenLM binary model (e.g., 3gram.bin). If provided, LM fusion is used.")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for LM fusion")
    parser.add_argument("--alpha", type=float, default=0.5, help="LM weight for shallow fusion")
    args = parser.parse_args()

    print(f"Prediction for video: {args.video_path}")
    greedy_pred, fused_pred = predict_lip_reading(args.video_path, args.checkpoint, kenlm_model_path=args.kenlm_model, beam_size=args.beam_size, alpha=args.alpha)
    print("Predicted (greedy):", greedy_pred)
    if args.kenlm_model is not None:
        print("Predicted (with LM fusion):", fused_pred)
