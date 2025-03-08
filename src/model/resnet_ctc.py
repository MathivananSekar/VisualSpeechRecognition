import torch
import torch.nn as nn
import torchvision.models.video as models
from torchvision.models.video import R3D_18_Weights
import math

############################
# Positional Encoding
############################
class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for batch-first inputs: (batch, time, d_model)."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # pe will be of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Sine for even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd dims

        # Register as a buffer so it's not trainable
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, time, d_model)
        """
        seq_len = x.size(1)  # the 'time' dimension
        # slice [0 : seq_len] -> shape (seq_len, d_model)
        pos_embed = self.pe[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        return x + pos_embed  # broadcast over batch

############################
# LipReadingModel (Encoder-Only + CTC)
############################
class LipReadingModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        """
        Encoder-only model for lip reading with CTC.
        - 3D ResNet backbone
        - Linear to reduce dimension (512 -> hidden_dim)
        - PositionalEncoding
        - TransformerEncoder (batch_first=True)
        - Final classification layer -> vocab_size
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Load the 3D ResNet
        full_r3d = models.r3d_18(weights=R3D_18_Weights.DEFAULT)
        
        # Remove the last two layers (avgpool + fc) to keep (B, 512, T', H', W')
        self.backbone = nn.Sequential(*list(full_r3d.children())[:-2])
        
        # Map 512 channels -> hidden_dim
        self.reduce_dim = nn.Linear(512, hidden_dim)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            batch_first=True  # We'll keep (batch, time, dim) shape
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Final classification layer (vocab_size includes blank token for CTC)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x shape: (batch, 3, T, H, W) if RGB (or (batch, 1, T, H, W) if grayscale, 
                 but repeated to 3 channels externally).
        Returns: (time, batch, vocab_size) for CTC loss.
        """
        # 1) Pass input through 3D ResNet backbone:
        #    shape: (B, 512, T', H', W')
        features = self.backbone(x)

        # 2) Unpack shape to flatten the spatial dims
        B, C, Tprime, Hprime, Wprime = features.shape
        TprimeHWprime = Tprime * Hprime * Wprime

        # 3) Flatten spatial+time dims: (B, 512, T'*H'*W') -> (B, C, TprimeHWprime)
        features = features.view(B, C, TprimeHWprime)

        # 4) Permute to batch-first, time-later shape: (B, TprimeHWprime, C)
        features = features.permute(0, 2, 1)

        # 5) Reduce dimension: 512 -> hidden_dim
        features = self.reduce_dim(features)  # (B, TprimeHWprime, hidden_dim)

        # 6) Apply positional encoding
        features = self.pos_encoding(features)  # (B, TprimeHWprime, hidden_dim)

        # 7) Transformer encoder (batch_first=True)
        encoded = self.encoder(features)        # (B, TprimeHWprime, hidden_dim)

        # 8) Final classification -> (B, TprimeHWprime, vocab_size)
        logits = self.classifier(encoded)

        # 9) For CTC, shape must be (time, batch, vocab_size):
        logits = logits.permute(1, 0, 2)        # (TprimeHWprime, B, vocab_size)
        return logits
