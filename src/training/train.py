import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.training.dataset import LipReadingDataset
from src.model.resnet_ctc import LipReadingModel
from src.training.vocab import vocab_size
from src.training.collate_fn import collate_fn

# --------------------
# Hyperparameters
# --------------------
BATCH_SIZE = 16   # Adjust as needed for GPU memory
EPOCHS = 50
LR = 1e-4
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Gather ALL .npy files for s1..s34
# --------------------
data_base = "data/processed"
speakers = [f"s{i}" for i in range(1, 35)]  # s1 to s34

npy_files = []
for spk in speakers:
    spk_npy_dir = os.path.join(data_base, spk, "npy")
    if os.path.isdir(spk_npy_dir):
        for fname in os.listdir(spk_npy_dir):
            if fname.endswith(".npy"):
                full_path = os.path.join(spk_npy_dir, fname)
                npy_files.append(full_path)

if not npy_files:
    raise ValueError("No .npy files found for any speakers in data/processed/sX/npy.")

# --------------------
# Create Dataset & DataLoader
# --------------------
dataset = LipReadingDataset(npy_files)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# --------------------
# Initialize Model & Loss
# --------------------
model = LipReadingModel(vocab_size=vocab_size()).to(DEVICE)
ctc_loss = nn.CTCLoss(blank=0)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# --------------------
# Training Loop
# --------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for frames, labels_list in dataloader:
        # frames shape: (B, 1, T, H, W)
        frames = frames.to(DEVICE)

        optimizer.zero_grad()

        # Repeat the single channel -> (B, 3, T, H, W)
        frames = frames.repeat(1, 3, 1, 1, 1)

        # Forward pass -> (T, B, vocab_size)
        logits = model(frames)
        log_probs = F.log_softmax(logits, dim=-1)
        time_dim = log_probs.shape[0]

        # Prepare lengths
        batch_size, _, max_frames, _, _ = frames.shape
        input_lengths = torch.full(
            (batch_size,),
            fill_value=time_dim,
            dtype=torch.long
        ).to(DEVICE)

        # Flatten variable-length labels
        target_labels = []
        label_lengths = []
        for label_tensor in labels_list:
            target_labels.append(label_tensor)
            label_lengths.append(label_tensor.size(0))

        target_labels = torch.cat(target_labels).to(DEVICE)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long).to(DEVICE)

        # Compute CTC loss
        loss = ctc_loss(log_probs, target_labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

print("Training completed!")
