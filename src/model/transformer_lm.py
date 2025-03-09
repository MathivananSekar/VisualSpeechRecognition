import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse

############################################
# 1. Build Vocabulary for LM
############################################
class Vocab:
    """
    A simple word-level vocabulary for the language model.
    Special tokens: <unk>, <pad>, <s>, </s>
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.unk_token = "<unk>"
        self.add_word(self.unk_token)  # index 0 for <unk>
        self.pad_token = "<pad>"
        self.add_word(self.pad_token)  # index 1 for <pad>
        self.start_token = "<s>"
        self.add_word(self.start_token)  # index 2 for <s>
        self.end_token = "</s>"
        self.add_word(self.end_token)    # index 3 for </s>

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)

    def word2id(self, word):
        return self.word2idx.get(word, 0)  # 0 is <unk> index

    def id2word(self, idx):
        if 0 <= idx < len(self.idx2word):
            return self.idx2word[idx]
        return self.unk_token

############################################
# 2. Dataset: Reading lines, tokenizing, and chunking into sequences
############################################
class TextDataset(Dataset):
    """
    Reads a text file (one sentence per line), tokenizes each line by whitespace,
    adds <s> and </s> tokens, concatenates all tokens, and then splits the token
    stream into sub-sequences of fixed length (seq_len). Each sub-sequence (of length seq_len)
    is paired with the next-token targets (also of length seq_len).
    """
    def __init__(self, file_path, vocab: Vocab, seq_len=20):
        self.vocab = vocab
        self.seq_len = seq_len
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Tokenize each line, adding <s> and </s>
        token_ids = []
        for line in lines:
            words = line.strip().split()
            # Add start and end tokens
            tokens = [self.vocab.start_token] + words + [self.vocab.end_token]
            for w in tokens:
                token_ids.append(self.vocab.word2id(w))
        # Convert list of token IDs to numpy array
        self.tokens = np.array(token_ids, dtype=np.int64)

    def __len__(self):
        # Ensure that we have a full sequence for both x and y.
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        # For sequence index idx, use tokens from idx*seq_len to idx*seq_len+seq_len as input,
        # and tokens from idx*seq_len+1 to idx*seq_len+seq_len+1 as target.
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]          # Input sequence
        y = self.tokens[start + 1 : start + self.seq_len + 1]    # Target sequence (next token)
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

############################################
# 3. Positional Encoding (batch-first)
############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

############################################
# 4. Transformer LM Model
############################################
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embed(x)             # (batch, seq_len, d_model)
        emb = self.pos_encoder(emb)       # (batch, seq_len, d_model)
        out = self.transformer(emb)       # (batch, seq_len, d_model)
        logits = self.fc_out(out)         # (batch, seq_len, vocab_size)
        return logits

############################################
# 5. Training / Evaluation Loop
############################################
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)  # x, y: (batch, seq_len)
        optimizer.zero_grad()
        logits = model(x)  # (batch, seq_len, vocab_size)
        batch_size, seq_len, vocab_sz = logits.shape
        loss = criterion(logits.view(-1, vocab_sz), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            batch_size, seq_len, vocab_sz = logits.shape
            loss = criterion(logits.view(-1, vocab_sz), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

############################################
# 6. Main Script
############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="grid_train.txt", help="Training corpus (one utterance per line)")
    parser.add_argument("--val_file", type=str, default=None, help="Validation corpus (optional)")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------
    # 1) Build vocabulary from train (and optionally val) files
    # --------------------------
    vocab = Vocab()

    def add_file_to_vocab(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.strip().split():
                    vocab.add_word(w)

    add_file_to_vocab(args.train_file)
    if args.val_file is not None and os.path.exists(args.val_file):
        add_file_to_vocab(args.val_file)

    print(f"Vocabulary size: {len(vocab)}")

    # --------------------------
    # 2) Create Datasets & DataLoaders
    # --------------------------
    train_dataset = TextDataset(args.train_file, vocab, seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_file is not None and os.path.exists(args.val_file):
        val_dataset = TextDataset(args.val_file, vocab, seq_len=args.seq_len)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --------------------------
    # 3) Create Model
    # --------------------------
    model = TransformerLM(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.pad_token])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------
    # 4) Training Loop
    # --------------------------
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PPL: {math.exp(val_loss):.2f}")
        else:
            print(f"Epoch {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    main()
