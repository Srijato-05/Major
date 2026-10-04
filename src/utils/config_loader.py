import os
import yaml
from pathlib import Path
from typing import Any, Dict

def get_project_root() -> Path:
    """Returns the absolute root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent

DEFAULT_FALLBACKS: Dict[str, Any] = {
    "system": {
        "project_name": "adaptive-multimodal-ewaste-sorting",
        "version": "1.0.0",
        "random_seed": 42,
        "device": "cuda"
    },
    "conveyor": {
        "speed_mps": 2.5,
        "default_speed_mps": 2.5,
        "min_speed_mps": 1.0,
        "max_speed_mps": 4.0,
        "speed_range_mps": [1.0, 4.0],
        "belt_width_mm": 1000.0,
        "inspection_length_mm": 1500.0,
        "sampling_rate_fps": 60,
        "conveyor_direction": "downstream",
        "belt_friction_coefficient": 0.65,
        "belt_slip_noise_std": 0.02
    },
    "optical_geometry": {
        "scale_x_mm_per_px": 1.5625,
        "scale_y_mm_per_px": 1.5625,
        "origin_offset_x_mm": 0.0,
        "origin_offset_y_mm": 0.0,
        "fov_width_mm": 1000.0,
        "fov_length_mm": 1200.0,
        "camera_mounting_height_mm": 850.0
    },
    "homography": {
        "scale_x_mm_per_px": 1.5625,
        "scale_y_mm_per_px": 1.5625,
        "origin_offset_x_mm": 0.0,
        "origin_offset_y_mm": 0.0
    },
    "model": {
        "architecture": "multimodal_4ch_deeplabv3",
        "num_classes": 6,
        "in_channels": 4,
        "img_size": 640,
        "conf_threshold": 0.35,
        "iou_threshold": 0.50,
        "max_det": 100
    },
    "training": {
        "batch_size": 4,
        "stage1_freeze_epochs": 2,
        "stage1_lr": 0.001,
        "stage2_unfreeze_epochs": 5,
        "stage2_lr_max": 0.0001,
        "stage2_lr_min": 0.000001,
        "weight_decay": 0.0005,
        "loss_weights": {
            "lambda_cls": 0.5,
            "lambda_mask": 2.5,
            "lambda_box": 7.5,
            "lambda_dfl": 1.5
        },
        "augmentations": {
            "blur_prob": 0.5,
            "glare_prob": 0.4,
            "dust_prob": 0.3,
            "cutout_prob": 0.3,
            "angular_jitter_deg": 2.0,
            "max_cutout_ratio": 0.30
        }
    },
    "tracking": {
        "tracker_type": "bytetrack",
        "track_thresh": 0.5,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "min_box_area": 50
    },
    "telemetry": {
        "window_seconds": 10.0,
        "cost_electricity_eur_kwh": 0.22,
        "conveyor_power_kw": 4.5
    }
}

def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges dictionary `update` into `base` without losing defaults."""
    res = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and k in res and isinstance(res[k], dict):
            res[k] = deep_merge(res[k], v)
        else:
            res[k] = v
    return res

def load_config(system_config_path: str = None, conveyor_config_path: str = None) -> Dict[str, Any]:
    """
    Loads and merges system and conveyor configuration files with guaranteed safe fallbacks.
    Even if config files are missing, partially empty, or missing keys, default values prevail.
    """
    root = get_project_root()
    if system_config_path is None:
        system_config_path = os.path.join(root, "configs", "system_config.yaml")
    if conveyor_config_path is None:
        conveyor_config_path = os.path.join(root, "configs", "conveyor_config.yaml")

    config = dict(DEFAULT_FALLBACKS)

    # Load system_config.yaml if available
    if os.path.exists(system_config_path):
        try:
            with open(system_config_path, "r", encoding="utf-8") as f:
                sys_cfg = yaml.safe_load(f) or {}
                config = deep_merge(config, sys_cfg)
        except Exception as e:
            print(f"Warning: Failed to parse system_config.yaml ({e}), using safe defaults.")

    # Load conveyor_config.yaml if available (overrides/extends conveyor parameters)
    if os.path.exists(conveyor_config_path):
        try:
            with open(conveyor_config_path, "r", encoding="utf-8") as f:
                conv_cfg = yaml.safe_load(f) or {}
                config = deep_merge(config, conv_cfg)
        except Exception as e:
            print(f"Warning: Failed to parse conveyor_config.yaml ({e}), using safe defaults.")

    # Ensure speed_mps and default_speed_mps stay synchronized
    speed = config.get("conveyor", {}).get("speed_mps") or config.get("conveyor", {}).get("default_speed_mps", 2.5)
    config["conveyor"]["speed_mps"] = speed
    config["conveyor"]["default_speed_mps"] = speed

    return config

# Global singleton config accessor
CONFIG = load_config()

