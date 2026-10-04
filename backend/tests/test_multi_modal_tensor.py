"""
Unit Test: Multi-Modal 4-Channel Tensor Construction
"""

import pytest
import numpy as np
import torch
from data.multi_modal_loader import create_4channel_tensor

def test_multi_modal_tensor_shape():
    rgb = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    nir = np.random.randint(0, 255, (640, 640), dtype=np.uint8)

    tensor = create_4channel_tensor(rgb, nir)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (4, 640, 640)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
