import os
import sys
import json
import yaml
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.config_loader import CONFIG
from src.data.multimodal_dataset import build_multimodal_dataloader
from src.models.multimodal_segmenter import build_4channel_segmentation_model
from src.models.evaluation_suite import ModelEvaluationSuite


def apply_class_thresholds(probs: torch.Tensor, thresholds: dict) -> torch.Tensor:
    """
    Applies calibrated foreground probability thresholds to softmax probs [B, C, H, W].
    Pixels whose highest foreground probability meets its class threshold take that class;
    otherwise they fall back to Class 0 (Conveyor Background).
    """
    fg_probs = probs[:, 1:, :, :]
    fg_max, fg_idx = torch.max(fg_probs, dim=1)
    fg_cls = fg_idx + 1  # Shift [0..5] -> [1..6]

    thresh_map = torch.empty_like(fg_max)
    for c in range(1, probs.shape[1]):
        thresh_map[fg_cls == c] = float(thresholds.get(c, 0.45))

    return torch.where(fg_max >= thresh_map, fg_cls, torch.zeros_like(fg_cls))


def calibrate_on_validation(weights_path: str, img_size: int = 384, batch_size: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_4channel_segmentation_model(num_classes=7, in_channels=4, pretrained=False)
    chk = torch.load(weights_path, map_location=device, weights_only=False)
    sd = chk.get("model_state_dict", chk)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()

    val_loader = build_multimodal_dataloader(
        split="val",
        batch_size=batch_size,
        img_size=(img_size, img_size),
        is_training=False,
        num_workers=0,
    )

    num_classes = 7
    num_bins = 100
    tp_hist = np.zeros((num_classes, num_bins), dtype=np.int64)
    fp_hist = np.zeros((num_classes, num_bins), dtype=np.int64)
    gt_totals = np.zeros(num_classes, dtype=np.int64)
    raw_suite = ModelEvaluationSuite(num_classes=num_classes)

    print(f"Pass 1/2: Sweeping probability histograms across {len(val_loader.dataset)} validation samples...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calibrate Hist"):
            images = batch["image"].to(device)
            targets = batch["mask"].numpy()
            logits = model(images)["out"]
            probs = F.softmax(logits, dim=1)

            raw_preds = torch.argmax(probs, dim=1).cpu().numpy()
            raw_suite.update(raw_preds, targets)

            fg_max, fg_idx = torch.max(probs[:, 1:, :, :], dim=1)
            fg_cls = (fg_idx + 1).cpu().numpy()
            fg_max_np = fg_max.cpu().numpy()

            for c in range(1, num_classes):
                gt_c = (targets == c)
                gt_totals[c] += int(gt_c.sum())
                sel_c = (fg_cls == c)
                if not sel_c.any():
                    continue
                tp_probs = fg_max_np[sel_c & gt_c]
                fp_probs = fg_max_np[sel_c & (~gt_c)]
                if tp_probs.size > 0:
                    tp_hist[c] += np.histogram(tp_probs, bins=num_bins, range=(0.0, 1.0))[0]
                if fp_probs.size > 0:
                    fp_hist[c] += np.histogram(fp_probs, bins=num_bins, range=(0.0, 1.0))[0]

    raw_metrics = raw_suite.compute_metrics()

    # Find IoU-maximizing threshold t in [0.20, 0.95] per foreground class
    calibrated = {}
    for c in range(1, num_classes):
        tp_cum = np.cumsum(tp_hist[c][::-1])[::-1]
        fp_cum = np.cumsum(fp_hist[c][::-1])[::-1]
        gt_c = max(int(gt_totals[c]), 1)

        best_t = 0.50
        best_iou = -1.0
        for b_idx in range(20, 96):
            tp = int(tp_cum[b_idx])
            fp = int(fp_cum[b_idx])
            fn = gt_c - tp
            rec = tp / gt_c
            # Safety constraint: keep Battery Pack recall >= 90%
            if c == 3 and rec < 0.90:
                continue
            denom = tp + fp + fn
            iou = (tp / denom) if denom > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_t = round(b_idx / 100.0, 2)

        calibrated[c] = best_t

    # Pass 2: Verify exact calibrated validation metrics
    cal_suite = ModelEvaluationSuite(num_classes=num_classes)
    print(f"Pass 2/2: Verifying calibrated thresholds {calibrated} on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Verify Val"):
            images = batch["image"].to(device)
            targets = batch["mask"].numpy()
            probs = F.softmax(model(images)["out"], dim=1)
            preds = apply_class_thresholds(probs, calibrated).cpu().numpy()
            cal_suite.update(preds, targets)

    cal_metrics = cal_suite.compute_metrics()

    print("\n" + "=" * 78)
    print(f"[VAL CALIBRATION] Raw mIoU={raw_metrics['mean_iou']*100:.2f}% (OA={raw_metrics['overall_pixel_accuracy']*100:.2f}%) "
          f"-> Calibrated mIoU={cal_metrics['mean_iou']*100:.2f}% (OA={cal_metrics['overall_pixel_accuracy']*100:.2f}%)")
    print(f"Optimal per-class thresholds: {calibrated}")
    print("=" * 78)

    out_json = "models/checkpoints/calibrated_thresholds.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "weights_path": weights_path,
            "img_size": img_size,
            "class_thresholds": {str(k): v for k, v in calibrated.items()},
            "val_raw": raw_metrics,
            "val_calibrated": cal_metrics,
        }, f, indent=2)

    cfg_path = "configs/system_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        sys_cfg = yaml.safe_load(f) or {}
    sys_cfg.setdefault("model", {})["class_thresholds"] = calibrated
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sys_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"Saved calibrated thresholds to {out_json} and updated {cfg_path}.")
    return calibrated


if __name__ == "__main__":
    default_img_size = int(CONFIG.get("model", {}).get("img_size", 384))
    parser = argparse.ArgumentParser(description="Calibrate per-class probability thresholds on val.json.")
    parser.add_argument("--weights", default="models/checkpoints/multimodal_4ch_best.pth")
    parser.add_argument("--img-size", type=int, default=default_img_size)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    calibrate_on_validation(args.weights, args.img_size, args.batch_size)

