"""
Multi-Modal Loader
Constructs 4-channel input tensors (RGB + NIR Band / Emissivity) for PyTorch / TensorRT inference pipelines.
"""

import numpy as np
import torch

def create_4channel_tensor(rgb_image: np.ndarray, nir_band: np.ndarray) -> torch.Tensor:
    """
    Combines 3-channel RGB image (H, W, 3) and 1-channel NIR band (H, W) into 4-channel tensor (4, H, W).
    """
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
        if nir_band.ndim == 2:
            nir_band = np.expand_dims(nir_band, axis=2)
        combined = np.concatenate([rgb_image, nir_band], axis=2)
        tensor = torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0
        return tensor
    raise ValueError("Invalid RGB or NIR input dimensions.")
