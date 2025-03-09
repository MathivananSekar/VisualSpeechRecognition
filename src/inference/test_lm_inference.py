import os
import torch
import torch.nn.functional as F
import argparse

# Import your Transformer LM model and Vocab from your LM-specific scripts.
from src.model.transformer_lm import TransformerLM,Vocab

def generate_text(model, vocab, prompt, max_len=50, device="cpu"):
    """
    Generates text from a given prompt using greedy decoding.
    
    Args:
        model: Trained TransformerLM model.
        vocab: An instance of Vocab with the LM vocabulary.
        prompt: Input prompt string.
        max_len: Maximum number of tokens to generate.
        device: Device for inference.
        
    Returns:
        A string containing the generated text.
    """
    # Convert prompt to token IDs: start with <s> then prompt words.
    prompt_tokens = [vocab.word2id(vocab.start_token)] + [vocab.word2id(w) for w in prompt.strip().split()]
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)  # shape: (1, L)
    
    model.eval()
    generated = input_ids  # (1, current_seq_len)
    with torch.no_grad():
        for _ in range(max_len):
            # Forward pass: (batch, seq_len, vocab_size)
            logits = model(generated)
            # Take logits for the last token
            last_logits = logits[:, -1, :]
            # Greedy: choose token with highest log probability
            next_token = torch.argmax(F.log_softmax(last_logits, dim=-1), dim=-1).unsqueeze(0)
            if next_token.item() == vocab.word2id(vocab.end_token):
                break
            generated = torch.cat([generated, next_token], dim=1)
    
    generated_ids = generated[0].tolist()
    # Convert token IDs back to words and remove special tokens.
    output_words = [vocab.id2word(tok) for tok in generated_ids if tok not in [
        vocab.word2id(vocab.start_token),
        vocab.word2id(vocab.end_token),
        vocab.word2id(vocab.pad_token)
    ]]
    return " ".join(output_words)

def main():
    parser = argparse.ArgumentParser(description="Test inference for Transformer LM")
    parser.add_argument("--prompt", type=str, default="place red at", help="Input prompt text")
    parser.add_argument("--checkpoint", type=str, default="lm_checkpoint.pth", help="Path to LM model checkpoint")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--train_file", type=str, default="grid_train.txt", help="Path to training corpus (for vocab)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build vocabulary from the training file to match the training vocabulary.
    vocab = Vocab()
    if os.path.exists(args.train_file):
        with open(args.train_file, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.strip().split():
                    vocab.add_word(w)
    else:
        print(f"Training file {args.train_file} not found; using default vocab.")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create the Transformer LM model with the built vocabulary size.
    model = TransformerLM(
        vocab_size=len(vocab),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    ).to(device)
    
    # Load the trained checkpoint.
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Exiting.")
        return
    
    # Generate text from the given prompt.
    generated_text = generate_text(model, vocab, args.prompt, max_len=args.max_len, device=device)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()
