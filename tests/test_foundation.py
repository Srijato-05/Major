import pytest
import numpy as np
import torch
import torch.nn as nn
from src.utils.config_loader import CONFIG, load_config
from src.models.weight_inflation import inflate_first_conv_weights
from src.kinematics.homography import PlanarHomography
from src.sorting.decision_engine import SortingDecisionEngine
from src.telemetry.telemetry_engine import IndustrialTelemetryEngine

def test_config_loader():
    config = load_config()
    assert "conveyor" in config
    assert "taxonomy" in config
    assert config["conveyor"]["default_speed_mps"] == 2.5
    assert len(config["taxonomy"]["classes"]) == 6

def test_weight_inflation():
    conv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    conv4 = inflate_first_conv_weights(conv3, target_in_channels=4, mode="mean_rgb")
    assert conv4.in_channels == 4
    assert conv4.out_channels == 32

    # Verify 4th channel is the mean of the first 3
    expected_mean = conv3.weight.data.mean(dim=1, keepdim=True)
    assert torch.allclose(conv4.weight.data[:, 3:4, :, :], expected_mean)

    # Test forward pass with dummy 4-channel tensor
    x = torch.randn(2, 4, 64, 64)
    out = conv4(x)
    assert out.shape == (2, 32, 64, 64)

def test_planar_homography():
    homog = PlanarHomography(scale_x_mm_per_px=2.0, scale_y_mm_per_px=2.0)
    xw, yw = homog.pixel_to_world(50, 100)
    assert xw == 100.0
    assert yw == 200.0

    u, v = homog.world_to_pixel(100.0, 200.0)
    assert abs(u - 50.0) < 1e-4
    assert abs(v - 100.0) < 1e-4

    # Test polygon area (square of 100x100 px = 200x200 mm = 40,000 mm^2)
    poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
    area = homog.polygon_area_mm2(poly)
    assert area == 40000.0

def test_sorting_decision_engine():
    engine = SortingDecisionEngine(CONFIG["taxonomy"])
    item = engine.evaluate_item(track_id=1, class_id=0, confidence=0.95, area_mm2=1000.0)
    assert item.class_name == "High-Grade PCB"
    assert item.target_bin == "BIN_PCB_RECOVERY"
    assert item.commodity_value_eur > 0
    assert item.estimated_mass_kg > 0

def test_telemetry_engine():
    telem = IndustrialTelemetryEngine(window_seconds=5.0)
    engine = SortingDecisionEngine(CONFIG["taxonomy"])
    item = engine.evaluate_item(track_id=1, class_id=1, confidence=0.9, area_mm2=500.0)
    
    telem.record_item(item)
    telem.record_latency(12.0)
    metrics = telem.get_metrics()

    assert metrics["total_items_sorted"] == 1
    assert metrics["avg_latency_ms"] == 12.0
    assert metrics["fps"] > 0
    assert "BIN_IC_RECOVERY" in metrics["bin_breakdown"]
