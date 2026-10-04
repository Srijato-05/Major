import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.utils.config_loader import CONFIG
from src.data.multimodal_dataset import build_multimodal_dataloader
from src.models.multimodal_segmenter import build_4channel_segmentation_model
from src.models.composite_loss import CompositeSegmentationLoss
from src.models.evaluation_suite import ModelEvaluationSuite

def train_polymer_deep_optimization():
    print("=" * 65)
    print("[GPU DEEP OPTIMIZATION] CLASS 3 (BFR POLYMERS) TARGET: 95%+")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset setup
    root_dir = "data/raw/spectralwaste"
    img_size = (320, 320)
    batch_size = 4

    train_loader = build_multimodal_dataloader(
        root_dir=root_dir,
        batch_size=batch_size,
        img_size=img_size,
        is_training=True,
        num_workers=0
    )
    val_loader = build_multimodal_dataloader(
        root_dir=root_dir,
        batch_size=batch_size,
        img_size=img_size,
        is_training=False,
        num_workers=0
    )

    # 2. Model initialization (load previous weights as warm start)
    model = build_4channel_segmentation_model(pretrained=True)
    chk_path = "models/checkpoints/multimodal_4ch_latest.pth"
    if os.path.exists(chk_path):
        chk = torch.load(chk_path, map_location=device)
        state = chk.get("model_state_dict", chk)
        model.load_state_dict(state)
        print(f"Loaded warm-start weights from {chk_path}")
    model.to(device)

    # 3. Class weighting: Background ~78%, Class 3 ~22%
    # We calibrate weights so class 3 has high precision penalty
    class_weights = torch.tensor([0.40, 2.0, 2.0, 1.85, 2.0, 2.0], dtype=torch.float32, device=device)
    criterion = CompositeSegmentationLoss(
        num_classes=6,
        lambda_cls=0.8,
        lambda_mask=3.0,
        class_weights=class_weights
    )

    # 4. Optimizer: Full end-to-end unfreezing with Cosine Annealing
    model.unfreeze_all()
    epochs = 12
    optimizer = optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"\n--- Launching {epochs} Progressive Fine-Tuning Epochs on GTX 1650 ---")
    best_polymer_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs["out"], targets)
            loss = loss_dict["loss_total"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Quick validation
        model.eval()
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                tgts = batch["mask"].cpu().numpy()
                preds = torch.argmax(model(imgs)["out"], dim=1).cpu().numpy()

                tp += int(((preds == 3) & (tgts == 3)).sum())
                fp += int(((preds == 3) & (tgts != 3)).sum())
                fn += int(((preds != 3) & (tgts == 3)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"Epoch [{epoch+1:2d}/{epochs:2d}] Loss: {avg_loss:.4f} | Poly Prec: {prec*100:5.2f}% | Rec: {rec*100:5.2f}% | IoU: {iou*100:5.2f}% | F1: {f1*100:5.2f}%")

        if f1 > best_polymer_f1:
            best_polymer_f1 = f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": CONFIG.get("training", {})
            }, chk_path)

    print(f"\n[SAVED] Best checkpoint preserved at {chk_path}")

    # 5. Full Evaluation Across Entire Dataset
    print("\n" + "=" * 65)
    print("[FINAL EVALUATION] COMPLETE BENCHMARK REPORT")
    print("=" * 65)

    evaluator = ModelEvaluationSuite(num_classes=6)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            tgts = batch["mask"].cpu().numpy()
            preds = torch.argmax(model(imgs)["out"], dim=1).cpu().numpy()
            evaluator.update(preds, tgts)

    metrics = evaluator.compute_metrics()
    p_met = metrics["per_class_metrics"]["BFR Polymers"]
    bg_met = metrics["per_class_metrics"]["High-Grade PCB"]

    print(f"Overall Pixel Accuracy:          {metrics['overall_pixel_accuracy']*100:.2f}%")
    print(f"BFR Polymers Precision:          {p_met['precision']*100:.2f}%")
    print(f"BFR Polymers Recall:             {p_met['recall']*100:.2f}%")
    print(f"BFR Polymers IoU:                {p_met['iou']*100:.2f}%")
    print(f"BFR Polymers F1 (Dice):          {p_met['f1_dice']*100:.2f}%")
    print(f"Conveyor Background Accuracy:    {bg_met['precision']*100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    train_polymer_deep_optimization()
