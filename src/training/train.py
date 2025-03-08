import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.training.dataset import LipReadingDataset
from src.model.resnet_ctc import LipReadingModel
from src.training.vocab import VOCAB, vocab_size
from src.training.collate_fn import collate_fn

# Hyperparameters
BATCH_SIZE = 2   # For testing, set small. Increase if GPU can handle more
EPOCHS = 10
LR = 1e-4
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data_path = "data/processed/s1/npy"
npy_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
dataset = LipReadingDataset(npy_files)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Initialize model
model = LipReadingModel(vocab_size=vocab_size()).to(DEVICE)
ctc_loss = nn.CTCLoss(blank=0)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for frames, labels_list in dataloader:

        # for lbl in labels_list:
        #     print("Min label:", lbl.min().item(), "Max label:", lbl.max().item())

        # frames shape: (B, 1, T, H, W)
        frames = frames.to(DEVICE)
        
        optimizer.zero_grad()
        
        # print("frames.shape =", frames.shape)

        # Repeat the single channel -> (B, 3, T, H, W)
        frames = frames.repeat(1, 3, 1, 1, 1)
        # print("after repeat frames.shape =", frames.shape)

        # Forward pass -> (T, B, vocab_size)
        logits = model(frames)
        log_probs = F.log_softmax(logits, dim=-1)
        time_dim = log_probs.shape[0] 

        # Prepare lengths
        # input_lengths is the same for entire batch because we padded
        batch_size, _, max_frames, _, _ = frames.shape
        input_lengths = torch.full((batch_size,), fill_value=time_dim, dtype=torch.long).to(DEVICE)

        # For CTC, we flatten the labels
        # But each sample might have a different label length
        target_labels = []
        label_lengths = []
        for label_tensor in labels_list:
            target_labels.append(label_tensor)
            label_lengths.append(label_tensor.size(0))

        # Concatenate all labels
        target_labels = torch.cat(target_labels).to(DEVICE)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long).to(DEVICE)


        # print("log_probs.shape =", log_probs.shape)  # Expect (T, B, vocab_size)
        # print("input_lengths =", input_lengths)
        # print("label_lengths =", label_lengths)

        # CTC expects shape: (T, B, C)
        # We already have (T, B, C) as log_probs
        # So no need to permute

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