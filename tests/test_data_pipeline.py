import pytest
import torch
import numpy as np
from src.data.augmentations import IndustrialMultimodalAugmentation
from src.data.multimodal_dataset import SpectralWasteDataset, build_multimodal_dataloader

def test_industrial_augmentations():
    aug = IndustrialMultimodalAugmentation(belt_speed_mps=3.0, blur_prob=1.0, glare_prob=1.0, dust_prob=1.0, cutout_prob=1.0)
    img_4ch = np.full((256, 256, 4), 128, dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.uint8)

    res = aug(img_4ch, mask)
    out_img = res["image"]
    out_mask = res["mask"]

    assert out_img.shape == (256, 256, 4)
    assert out_mask.shape == (256, 256)
    assert out_img.dtype == np.uint8

def test_spectralwaste_dataset():
    ds = SpectralWasteDataset(root_dir="data/raw/spectralwaste", img_size=(320, 320), is_training=False)
    assert len(ds) > 0

    item = ds[0]
    assert "image" in item
    assert "mask" in item
    assert item["image"].shape == (4, 320, 320)
    assert item["mask"].shape == (320, 320)
    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.int64

def test_multimodal_dataloader():
    loader = build_multimodal_dataloader(root_dir="data/raw/spectralwaste", batch_size=2, img_size=(320, 320), is_training=False)
    batch = next(iter(loader))

    assert batch["image"].shape == (2, 4, 320, 320)
    assert batch["mask"].shape == (2, 320, 320)
