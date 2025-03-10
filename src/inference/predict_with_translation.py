import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import kenlm

# Import lip reading model, vocab, preprocessing functions
from src.model.resnet_ctc import LipReadingModel
from src.training.vocab import VOCAB, vocab_size
from src.preprocessing.lip_detection import detect_lips
from src.preprocessing.extract_frames import extract_frames
from src.model.language_translator import translate_en_to_fr, translate_en_to_es


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_video(frames, resize=(32, 64)):
    """
    Preprocess frames into consistent shape (grayscale -> 3 channels).
    Final shape: (1, 3, num_frames, 32, 64)
    """
    processed_list = []
    for frame in frames:
        # Resize each frame (cv2 expects (width, height))
        frame_resized = cv2.resize(frame, (resize[1], resize[0]))
        processed_list.append(frame_resized)
    processed = np.array(processed_list, dtype=np.uint8)  # (num_frames, 32, 64)
    processed = torch.tensor(processed, dtype=torch.float32)  # (num_frames, 32, 64)
    processed = processed.unsqueeze(0).unsqueeze(1)  # (1, 1, num_frames, 32, 64)
    processed = processed.repeat(1, 3, 1, 1, 1)         # (1, 3, num_frames, 32, 64)
    return processed

def greedy_decode(ctc_probs, blank_index=0):
    """
    Greedy decoding: at each time step, choose the token with highest probability,
    skipping duplicates and blank tokens.
    ctc_probs: (time, vocab_size)
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
    Convert numeric token indices back to words using VOCAB.
    """
    idx_to_token = {v: k for k, v in VOCAB.items()}
    words = [idx_to_token.get(i, "<unk>") for i in indices if idx_to_token.get(i, "<unk>") not in ["<blank>", "sil"]]
    return " ".join(words)

def kenlm_score(kenlm_model, sentence):
    """
    Get LM score from KenLM (convert log10 to natural log).
    """
    score = kenlm_model.score(sentence, bos=True, eos=True)
    return score * np.log(10)

def ctc_shallow_fusion_beam_search(ctc_log_probs, kenlm_model, idx_to_token, blank_idx=0, beam_size=3, alpha=0.5):
    """
    Beam search decoding with shallow fusion using KenLM.
    
    Args:
        ctc_log_probs: Tensor of shape (T, vocab_size) with CTC log probabilities (natural log).
        kenlm_model: Loaded KenLM model.
        idx_to_token: Dictionary mapping token index -> word.
        blank_idx: Index for blank token.
        beam_size: Beam size.
        alpha: LM weight.
        
    Returns:
        Best hypothesis as a list of token indices.
    """
    T, vocab_size = ctc_log_probs.shape
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
                    new_sentence = (beam["sentence"] + " " + word).strip()
                lm_score = kenlm_score(kenlm_model, new_sentence) if new_sentence != "" else 0.0
                combined_score = beam["score"] + ctc_score + alpha * lm_score
                new_beams.append({
                    "hyp": new_hyp,
                    "score": combined_score,
                    "sentence": new_sentence
                })
        new_beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)
        beams = new_beams[:beam_size]
    
    best_beam = beams[0]
    return best_beam["hyp"]

def predict_lip_reading(video_path, checkpoint_path, kenlm_model_path=None, beam_size=3, alpha=0.5):
    """
    Predict the transcript from a video using the lip reading model.
    Returns:
      - Greedy prediction (without LM fusion)
      - LM fusion prediction (if kenlm_model_path is provided)
    """
    # 1) Load the lip reading model
    model = LipReadingModel(vocab_size())
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2) Preprocess video frames
    raw_frames = extract_frames(video_path)
    lip_frames = detect_lips(raw_frames)
    input_tensor = preprocess_video(lip_frames)
    input_tensor = input_tensor.to(DEVICE)

    # 3) Forward pass to get CTC logits -> (T, 1, vocab_size)
    with torch.no_grad():
        logits = model(input_tensor)
        log_probs = F.log_softmax(logits, dim=-1)
    log_probs = log_probs.squeeze(1)  # (T, vocab_size)

    # Greedy decoding (without LM fusion)
    greedy_indices = greedy_decode(log_probs, blank_index=0)
    greedy_text = labels_to_text(greedy_indices)

    fused_text = greedy_text  # default to greedy if no LM fusion
    if kenlm_model_path is not None:
        kenlm_model = kenlm.Model(kenlm_model_path)
        idx_to_token = {v: k for k, v in VOCAB.items()}
        beam_indices = ctc_shallow_fusion_beam_search(log_probs, kenlm_model, idx_to_token, blank_idx=0, beam_size=beam_size, alpha=alpha)
        fused_text = labels_to_text(beam_indices)
    
    return greedy_text, fused_text

def translate_text(text, target_lang, translator_model, translator_tokenizer, device):
    """
    Translates English text to the target language using the provided translator model and tokenizer.
    target_lang: 'fr' for French, 'es' for Spanish, otherwise returns original text.
    """
    if target_lang.lower() == "en":
        return text
    elif target_lang.lower() == "fr":
        return translate_en_to_fr(text, translator_model, translator_tokenizer, device)
    elif target_lang.lower() == "es":
        return translate_en_to_es(text, translator_model, translator_tokenizer, device)
    else:
        print(f"Target language {target_lang} not supported. Returning original text.")
        return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict video using lip reading model with LM fusion and translation")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_50.pth", help="Path to lip reading model checkpoint")
    parser.add_argument("--kenlm_model", type=str, default=None, help="Path to KenLM binary model (e.g., 3gram.bin). If provided, LM fusion is used.")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for LM fusion")
    parser.add_argument("--alpha", type=float, default=0.5, help="LM weight for shallow fusion")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language for translation (en, fr, es)")
    # Additional arguments for translator if needed (optional)
    args = parser.parse_args()

    print(f"Prediction for video: {args.video_path}")
    greedy_pred, fused_pred = predict_lip_reading(args.video_path, args.checkpoint, kenlm_model_path=args.kenlm_model, beam_size=args.beam_size, alpha=args.alpha)
    
    print("Predicted (greedy):", greedy_pred)
    if args.kenlm_model is not None:
        print("Predicted (with LM fusion):", fused_pred)
    else:
        fused_pred = greedy_pred
    
    # Now integrate translation if target language is not English.
    if args.target_lang.lower() != "en":
        # Load the MBart model and tokenizer for translation.
        from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
        translator_model_name = "facebook/mbart-large-50-many-to-many-mmt"
        try:
            tokenizer = MBart50TokenizerFast.from_pretrained(translator_model_name, src_lang="en_XX", tgt_lang="fr_XX" if args.target_lang.lower()=="fr" else "es_XX",model_max_length=100)
        except Exception as e:
            from transformers import MBartTokenizer
            tokenizer = MBartTokenizer.from_pretrained(translator_model_name, src_lang="en_XX", tgt_lang="fr_XX" if args.target_lang.lower()=="fr" else "es_XX",model_max_length=100)
            print(f"Error loading fast tokenizer: {e}.")
            exit(1)
        try:
            translator_model = MBartForConditionalGeneration.from_pretrained(translator_model_name).to(DEVICE)
        except Exception as e:
            print(f"Error loading MBart model: {e}. Exiting.")
            exit(1)
        
        # Translate the LM fusion prediction
        if args.target_lang.lower() == "fr":
            translated = translate_en_to_fr(fused_pred, translator_model, tokenizer, DEVICE)
        elif args.target_lang.lower() == "es":
            translated = translate_en_to_es(fused_pred, translator_model, tokenizer, DEVICE)
        else:
            translated = fused_pred
    else:
        translated = fused_pred

    print("Final Predictions:")
    print("Without LM fusion:", greedy_pred)
    print("With LM fusion:", fused_pred)
    print("After Translation to :", translated)
