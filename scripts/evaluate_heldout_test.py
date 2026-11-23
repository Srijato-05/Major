import os
import sys
import json
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from src.utils.config_loader import CONFIG
from src.data.multimodal_dataset import build_multimodal_dataloader
from src.models.multimodal_segmenter import build_4channel_segmentation_model
from src.models.evaluation_suite import ModelEvaluationSuite
from scripts.calibrate_thresholds import apply_class_thresholds


def evaluate_split(model, loader, device, use_rgb_ablation: bool = False, class_thresholds=None, max_batches: int = None):
    suite = ModelEvaluationSuite(num_classes=7)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch["image"].to(device)
            if use_rgb_ablation:
                images = images.clone()
                images[:, 3:4, :, :] = images[:, :3, :, :].mean(dim=1, keepdim=True)
            targets = batch["mask"].cpu().numpy()
            logits = model(images)["out"]
            if class_thresholds:
                probs = F.softmax(logits, dim=1)
                preds = apply_class_thresholds(probs, class_thresholds).cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            suite.update(preds, targets)
    return suite.compute_metrics()


def main():
    default_img_size = int(CONFIG.get("model", {}).get("img_size", 384))
    parser = argparse.ArgumentParser(description="Evaluate trained 7-class model on untouched test.json split + calibrated thresholds + 3-ch ablation.")
    parser.add_argument("--weights", type=str, default="models/checkpoints/multimodal_4ch_best.pth")
    parser.add_argument("--img-size", type=int, default=default_img_size)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--out-report", type=str, default="models/checkpoints/test_benchmark_report.json")
    args = parser.parse_args()

    weights_path = args.weights
    if not os.path.exists(weights_path):
        fallback = "models/checkpoints/multimodal_4ch_latest.pth"
        if os.path.exists(fallback):
            weights_path = fallback

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_4channel_segmentation_model(num_classes=7, in_channels=4, pretrained=False)

    if os.path.exists(weights_path):
        chk = torch.load(weights_path, map_location=device, weights_only=False)
        sd = chk.get("model_state_dict", chk)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded checkpoint ({len(filtered)}/{len(model_sd)} tensors): {weights_path}")
    else:
        print(f"Warning: checkpoint not found at {weights_path}; evaluating randomly initialized heads.")

    model.to(device)
    test_loader = build_multimodal_dataloader(
        split="test",
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        is_training=False,
        num_workers=0,
    )

    raw_thresh = CONFIG.get("model", {}).get("class_thresholds", {})
    class_thresholds = {int(k): float(v) for k, v in raw_thresh.items()} if raw_thresh else None

    metrics_raw_4ch = evaluate_split(model, test_loader, device, use_rgb_ablation=False, class_thresholds=None, max_batches=args.max_batches)
    metrics_4ch = evaluate_split(model, test_loader, device, use_rgb_ablation=False, class_thresholds=class_thresholds, max_batches=args.max_batches) if class_thresholds else metrics_raw_4ch
    metrics_3ch = evaluate_split(model, test_loader, device, use_rgb_ablation=True, class_thresholds=class_thresholds, max_batches=args.max_batches)

    cal_gain = {
        "overall_pixel_accuracy_delta": round(metrics_4ch["overall_pixel_accuracy"] - metrics_raw_4ch["overall_pixel_accuracy"], 4),
        "mean_iou_delta": round(metrics_4ch["mean_iou"] - metrics_raw_4ch["mean_iou"], 4),
        "mean_f1_delta": round(metrics_4ch["mean_f1_score"] - metrics_raw_4ch["mean_f1_score"], 4),
    }

    ablation_delta = {
        "overall_pixel_accuracy_delta": round(metrics_4ch["overall_pixel_accuracy"] - metrics_3ch["overall_pixel_accuracy"], 4),
        "mean_iou_delta": round(metrics_4ch["mean_iou"] - metrics_3ch["mean_iou"], 4),
        "mean_f1_delta": round(metrics_4ch["mean_f1_score"] - metrics_3ch["mean_f1_score"], 4),
    }

    report = {
        "weights_path": weights_path,
        "img_size": args.img_size,
        "test_samples": len(test_loader.dataset),
        "class_thresholds": class_thresholds or {},
        "4ch_raw_argmax": metrics_raw_4ch,
        "4ch_rgb_swir": metrics_4ch,
        "3ch_rgb_ablation": metrics_3ch,
        "calibration_gain": cal_gain,
        "ablation_delta": ablation_delta,
    }

    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 90)
    print(f"[RAW ARGMAX TEST]         4-Ch OA={metrics_raw_4ch['overall_pixel_accuracy']*100:.2f}% mIoU={metrics_raw_4ch['mean_iou']*100:.2f}% mF1={metrics_raw_4ch['mean_f1_score']*100:.2f}%")
    if class_thresholds:
        print(f"[CALIBRATED TEST]         4-Ch OA={metrics_4ch['overall_pixel_accuracy']*100:.2f}% mIoU={metrics_4ch['mean_iou']*100:.2f}% mF1={metrics_4ch['mean_f1_score']*100:.2f}% "
              f"(dOA={cal_gain['overall_pixel_accuracy_delta']*100:+.2f}% dmIoU={cal_gain['mean_iou_delta']*100:+.2f}%)")
    print(f"[RGB ABLATION]            3-Ch OA={metrics_3ch['overall_pixel_accuracy']*100:.2f}% mIoU={metrics_3ch['mean_iou']*100:.2f}%")
    print(f"[4-CH vs 3-CH DELTA]      dOA={ablation_delta['overall_pixel_accuracy_delta']*100:+.2f}% dmIoU={ablation_delta['mean_iou_delta']*100:+.2f}%")
    print("-" * 90)
    for cname, pcm in metrics_4ch["per_class_metrics"].items():
        raw_iou = metrics_raw_4ch["per_class_metrics"][cname]["iou"]
        print(
            f"  Class {pcm['class_id']} ({cname:<21}): "
            f"IoU={pcm['iou']*100:5.2f}% (was {raw_iou*100:5.2f}%) | Prec={pcm['precision']*100:5.2f}% | "
            f"Rec={pcm['recall']*100:5.2f}% | F1={pcm['f1_dice']*100:5.2f}%"
        )
    print(f"\nReport saved to: {args.out_report}")
    print("=" * 90)


if __name__ == "__main__":
    main()
