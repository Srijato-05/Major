import pytest
import torch
import numpy as np
from src.models.multimodal_segmenter import Multimodal4ChSegmentationModel, build_4channel_segmentation_model
from src.models.composite_loss import CompositeSegmentationLoss, DiceLoss
from src.models.inference_engine import MultimodalInferenceEngine
from src.models.train_pipeline import MultimodalTrainer

def test_multimodal_segmenter_architecture():
    model = build_4channel_segmentation_model(num_classes=7, in_channels=4, pretrained=False)
    assert model.in_channels == 4
    assert model.num_classes == 7

    # Test forward pass with 4-channel tensor
    x = torch.randn(2, 4, 160, 160)
    out = model(x)
    assert "out" in out
    assert out["out"].shape == (2, 7, 160, 160)

def test_composite_loss():
    crit = CompositeSegmentationLoss(num_classes=7, lambda_cls=0.5, lambda_mask=2.5)
    logits = torch.randn(2, 7, 64, 64)
    targets = torch.randint(0, 7, (2, 64, 64))

    loss_dict = crit(logits, targets)
    assert "loss_total" in loss_dict
    assert "loss_ce" in loss_dict
    assert "loss_dice" in loss_dict
    assert loss_dict["loss_total"].item() > 0

def test_inference_engine_4ch_and_fallback():
    model = build_4channel_segmentation_model(num_classes=7, in_channels=4, pretrained=False)
    engine = MultimodalInferenceEngine(model=model, conf_threshold=0.2)

    # 1. Test 3-channel RGB with automatic synthetic NIR fallback
    rgb = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    res_rgb = engine.predict(rgb)
    assert "instances" in res_rgb
    assert "mask" in res_rgb
    assert res_rgb["mask"].shape == (200, 200)

    # 2. Test paired RGB + HSI
    hsi = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    res_hsi = engine.predict(rgb, hsi)
    assert "instances" in res_hsi
    assert res_hsi["mask"].shape == (200, 200)


def test_focal_composite_loss_and_trainer_eval(tmp_path):
    crit = CompositeSegmentationLoss(num_classes=7, lambda_cls=0.5, lambda_mask=2.5, focal_gamma=2.0)
    logits = torch.randn(2, 7, 32, 32)
    targets = torch.randint(0, 7, (2, 32, 32))
    loss_dict = crit(logits, targets)
    assert loss_dict["loss_total"].item() > 0

    model = build_4channel_segmentation_model(num_classes=7, in_channels=4, pretrained=False)
    trainer = MultimodalTrainer(model=model, device="cpu", save_dir=str(tmp_path), focal_gamma=2.0, use_amp=False)

    dummy_loader = [{"image": torch.randn(2, 4, 64, 64), "mask": torch.randint(0, 7, (2, 64, 64))}]
    val_metrics = trainer.evaluate_epoch(dummy_loader)
    assert "mean_iou" in val_metrics
    assert "overall_pixel_accuracy" in val_metrics
    assert "loss_total" in val_metrics
