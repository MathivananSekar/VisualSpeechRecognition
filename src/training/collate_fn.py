import torch

def collate_fn(batch):
    """
    Custom collate function for variable-length video frames.
    We pad frames to match the longest sequence in the batch.
    Labels remain a list of varying lengths (we'll handle them in the train loop).
    """
    frames_list, labels_list = zip(*batch)

    # Find max frame length
    max_frames = max(f.shape[0] for f in frames_list)

    # Dimensions for frames: (batch, max_frames, 1, H, W)
    batch_size = len(frames_list)
    _, _, H, W = frames_list[0].shape  # from the first sample

    padded_frames = torch.zeros(batch_size, max_frames, 1, H, W, dtype=torch.float32)

    for i, f in enumerate(frames_list):
        length = f.shape[0]
        padded_frames[i, :length] = f  # copy frames

    # Return frames as (batch, 1, frames, H, W) by permuting
    # because our model expects: (batch, 1, T, H, W)
    padded_frames = padded_frames.permute(0, 2, 1, 3, 4)  # -> (B, 1, max_frames, H, W)

    # labels_list is a tuple of 1D tensors; let's keep it as is for now
    return padded_frames, labels_list
