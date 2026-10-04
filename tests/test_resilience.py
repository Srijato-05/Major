import pytest
import numpy as np
import os
from src.utils.config_loader import load_config, DEFAULT_FALLBACKS
from src.models.multimodal_segmenter import build_4channel_segmentation_model
from src.models.inference_engine import MultimodalInferenceEngine

def test_conveyor_config_separation():
    # Verify load_config loads conveyor parameters cleanly
    cfg = load_config()
    assert "conveyor" in cfg
    assert "speed_mps" in cfg["conveyor"]
    assert cfg["conveyor"]["speed_mps"] == 2.5
    assert "optical_geometry" in cfg
    assert cfg["optical_geometry"]["scale_x_mm_per_px"] > 0

def test_missing_parameters_and_files_resilience():
    # Test loading with non-existent config files (must NOT crash, must return safe defaults)
    fallback_cfg = load_config(system_config_path="non_existent_sys.yaml", conveyor_config_path="non_existent_conv.yaml")
    assert fallback_cfg["conveyor"]["speed_mps"] == 2.5
    assert fallback_cfg["model"]["num_classes"] == 6
    assert fallback_cfg["system"]["project_name"] == "adaptive-multimodal-ewaste-sorting"

def test_pure_rgb_image_inference():
    # Model accepts 4 channels, but user might upload pure 3-channel RGB image
    model = build_4channel_segmentation_model(num_classes=6, in_channels=4, pretrained=False)
    engine = MultimodalInferenceEngine(model=model)

    pure_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = engine.predict_frame(pure_rgb)
    assert "instances" in res
    assert "mask" in res
    assert res["mask"].shape == (256, 256)

def test_video_frame_rendering_and_empty_frames():
    model = build_4channel_segmentation_model(num_classes=6, in_channels=4, pretrained=False)
    engine = MultimodalInferenceEngine(model=model)

    # Empty frame should return empty safe dict without error
    empty_frame = np.array([])
    res_empty = engine.predict_frame(empty_frame)
    assert res_empty["num_detections"] == 0

    # Test annotate_frame with standard video frame
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    dummy_pred = {
        "instances": [{
            "class_id": 0,
            "class_name": "High-Grade PCB",
            "confidence": 0.88,
            "bbox": [20, 20, 50, 50],
            "polygon": [[20, 20], [70, 20], [70, 70], [20, 70]]
        }]
    }
    annotated = engine.annotate_frame(frame, dummy_pred)
    assert annotated.shape == (200, 200, 3)
    assert annotated.dtype == np.uint8
