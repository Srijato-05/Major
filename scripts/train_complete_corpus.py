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

def train_complete_corpus():
    print("=" * 80)
    print("[RIGOROUS TRAINING] COMPLETE DUAL DATASET CORPUS (852 SPECTRALWASTE + 944 BATTERIES = 1,796 TRIPLETS)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    img_size = (320, 320)
    batch_size = 4

    print(f"\nBuilding unified DataLoaders for all 1,796 multimodal triplets...")
    train_loader = build_multimodal_dataloader(
        batch_size=batch_size,
        img_size=img_size,
        is_training=True,
        num_workers=0,
        include_batteries=True
    )
    val_loader = build_multimodal_dataloader(
        batch_size=batch_size,
        img_size=img_size,
        is_training=False,
        num_workers=0,
        include_batteries=True
    )
    print(f"Total batches per epoch: {len(train_loader)} (Total frames: {len(train_loader.dataset)})")

    # Load warm-start model
    model = build_4channel_segmentation_model(pretrained=True)
    chk_path = "models/checkpoints/multimodal_4ch_latest.pth"
    if os.path.exists(chk_path):
        chk = torch.load(chk_path, map_location=device)
        state = chk.get("model_state_dict", chk)
        model.load_state_dict(state, strict=False)
        print(f"Warm start loaded from {chk_path}")
    model.to(device)

    # Class weights balancing all 6 industrial classes:
    # 0: PCB/BG, 1: IC Molds, 2: Battery Packs (Hazardous), 3: Polymers, 4: Heat Sinks, 5: Copper Harnesses
    class_weights = torch.tensor([0.35, 2.0, 3.0, 1.8, 2.0, 2.2], dtype=torch.float32, device=device)
    criterion = CompositeSegmentationLoss(
        num_classes=6,
        lambda_cls=0.8,
        lambda_mask=3.0,
        class_weights=class_weights
    )

    model.unfreeze_all()
    epochs = 8
    optimizer = optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"\n--- Rigorous Training across 1,796 Triplets for {epochs} Epochs ---")
    best_score = 0.0

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

        # Quick validation sample across 80 batches
        model.eval()
        tp_poly, fp_poly, fn_poly = 0, 0, 0
        tp_bat, fp_bat, fn_bat = 0, 0, 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                tgts = batch["mask"].cpu().numpy()
                preds = torch.argmax(model(imgs)["out"], dim=1).cpu().numpy()

                tp_poly += int(((preds == 3) & (tgts == 3)).sum())
                fp_poly += int(((preds == 3) & (tgts != 3)).sum())
                fn_poly += int(((preds != 3) & (tgts == 3)).sum())

                tp_bat += int(((preds == 2) & (tgts == 2)).sum())
                fp_bat += int(((preds == 2) & (tgts != 2)).sum())
                fn_bat += int(((preds != 2) & (tgts == 2)).sum())

                val_count += 1
                if val_count >= 80:
                    break

        poly_f1 = (2 * tp_poly / (2 * tp_poly + fp_poly + fn_poly)) if (2 * tp_poly + fp_poly + fn_poly) > 0 else 0.0
        bat_rec = (tp_bat / (tp_bat + fn_bat)) if (tp_bat + fn_bat) > 0 else 0.0
        bat_prec = (tp_bat / (tp_bat + fp_bat)) if (tp_bat + fp_bat) > 0 else 0.0
        bat_f1 = (2 * tp_bat / (2 * tp_bat + fp_bat + fn_bat)) if (2 * tp_bat + fp_bat + fn_bat) > 0 else 0.0

        score = poly_f1 * 0.45 + bat_f1 * 0.55
        print(f"Epoch [{epoch+1:2d}/{epochs:2d}] Loss: {avg_loss:.4f} | Poly F1: {poly_f1*100:5.2f}% | Bat Prec: {bat_prec*100:5.2f}% | Bat Rec: {bat_rec*100:5.2f}% | Bat F1: {bat_f1*100:5.2f}%")

        if score > best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": CONFIG.get("training", {})
            }, chk_path)

    print(f"\n[CHECKPOINT SAVED] Rigorous model weights saved to {chk_path}")

    # Exhaustive evaluation across the complete validation dataset
    print("\n" + "=" * 80)
    print("[EVALUATION] EXHAUSTIVE BENCHMARK OVER COMPLETE 1,796-SAMPLE CORPUS")
    print("=" * 80)
    evaluator = ModelEvaluationSuite(num_classes=6)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            targets = batch["mask"].numpy()
            preds = torch.argmax(model(imgs)["out"], dim=1).cpu().numpy()
            evaluator.update(preds, targets)

    metrics = evaluator.compute_metrics()
    print(f"Total Pixels Evaluated:          {metrics['total_pixels_evaluated']:,}")
    print(f"Overall Pixel Accuracy:          {metrics['overall_pixel_accuracy']*100:.2f}%")
    print("-" * 80)
    for cname, pcm in metrics["per_class_metrics"].items():
        if pcm["total_ground_truth_pixels"] > 0:
            print(f"-- {cname:<28} (Class {pcm['class_id']}): Rec={pcm['recall']*100:5.2f}% | Prec={pcm['precision']*100:5.2f}% | IoU={pcm['iou']*100:5.2f}% | F1={pcm['f1_dice']*100:5.2f}% | GT Pixels={pcm['total_ground_truth_pixels']:,}")
    print("=" * 80)

if __name__ == "__main__":
    train_complete_corpus()
