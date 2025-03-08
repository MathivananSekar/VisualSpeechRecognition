import torch
import numpy as np
from torch.utils.data import Dataset
from src.training.vocab import text_to_labels 

class LipReadingDataset(Dataset):
    """Custom PyTorch Dataset for Lip Reading (CTC)"""
    
    def __init__(self, npy_files):
        self.data = [np.load(f, allow_pickle=True).item() for f in npy_files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # frames shape: (num_frames, height, width)
        frames = torch.tensor(sample["frames"], dtype=torch.float32).unsqueeze(1)  # (num_frames, 1, H, W)
        labels_str = sample["labels"]  # list of strings, e.g. ["sil", "lay", "green", ...]
        
        # Convert words to numeric labels
        labels_numeric = text_to_labels(labels_str)  # e.g. [2, 10, 4...]
        
        return frames, torch.tensor(labels_numeric, dtype=torch.long)