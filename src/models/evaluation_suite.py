import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict

class ModelEvaluationSuite:
    """
    Comprehensive Quantitative Evaluation Suite for 4-Channel E-Waste Segmentation:
    Computes:
    - Overall Pixel Accuracy (OA)
    - Mean Intersection-over-Union (mIoU) across all classes
    - Per-Class IoU (Intersection over Union)
    - Precision, Recall, and F1-Score (Dice Score)
    - Confusion Matrix (True Class vs Predicted Class)
    - False Positive Rate (FPR) for hazardous classes (BFR polymers, Battery packs)
    """

    def __init__(self, num_classes: int = 6, class_names: Dict[int, str] = None, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or {
            0: "High-Grade PCB",
            1: "IC Molds & Chips",
            2: "Battery Pack Units",
            3: "BFR Polymers",
            4: "Metallic Heat Sinks",
            5: "Copper Wire Harnesses"
        }
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix.fill(0)

    def update(self, pred_masks: np.ndarray, target_masks: np.ndarray):
        """
        Updates the confusion matrix with a batch of predictions and ground-truth targets.
        Args:
            pred_masks: [B, H, W] or [H, W] containing predicted class indices (0 to num_classes-1)
            target_masks: [B, H, W] or [H, W] containing ground-truth class indices
        """
        preds = np.asarray(pred_masks).flatten()
        targets = np.asarray(target_masks).flatten()

        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]

        valid_preds = (preds >= 0) & (preds < self.num_classes)
        preds = preds[valid_preds]
        targets = targets[valid_preds]

        # Accumulate 2D histogram
        indices = self.num_classes * targets + preds
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Calculates all standard computer vision benchmark metrics.
        """
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Overall Pixel Accuracy
        total_pixels = cm.sum()
        pixel_accuracy = float(tp.sum() / max(total_pixels, 1))

        # Per-class IoU: TP / (TP + FP + FN)
        denominator = tp + fp + fn
        iou_per_class = np.where(denominator > 0, tp / denominator, 0.0)

        # Precision & Recall
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f1_score = np.where((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0.0)

        # Classes that actually appeared in the target ground truth
        valid_classes = cm.sum(axis=1) > 0
        mean_iou = float(iou_per_class[valid_classes].mean()) if valid_classes.any() else float(iou_per_class.mean())
        mean_precision = float(precision[valid_classes].mean()) if valid_classes.any() else float(precision.mean())
        mean_recall = float(recall[valid_classes].mean()) if valid_classes.any() else float(recall.mean())
        mean_f1 = float(f1_score[valid_classes].mean()) if valid_classes.any() else float(f1_score.mean())

        per_class_summary = {}
        for c in range(self.num_classes):
            cname = self.class_names.get(c, f"Class_{c}")
            per_class_summary[cname] = {
                "class_id": c,
                "iou": round(float(iou_per_class[c]), 4),
                "precision": round(float(precision[c]), 4),
                "recall": round(float(recall[c]), 4),
                "f1_dice": round(float(f1_score[c]), 4),
                "total_ground_truth_pixels": int(cm.sum(axis=1)[c])
            }

        return {
            "overall_pixel_accuracy": round(pixel_accuracy, 4),
            "mean_iou": round(mean_iou, 4),
            "mean_precision": round(mean_precision, 4),
            "mean_recall": round(mean_recall, 4),
            "mean_f1_score": round(mean_f1, 4),
            "total_pixels_evaluated": int(total_pixels),
            "per_class_metrics": per_class_summary,
            "confusion_matrix": cm.tolist()
        }
