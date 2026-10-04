import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from src.utils.config_loader import CONFIG
from src.data.multimodal_dataset import build_multimodal_dataloader
from src.models.train_pipeline import MultimodalTrainer
from src.models.evaluation_suite import ModelEvaluationSuite
from src.models.inference_engine import MultimodalInferenceEngine

def run_gpu_training_job():
    print("=" * 60)
    print("[GPU TRAIN] INITIALIZING MULTIMODAL TRAINING ON NVIDIA GTX 1650")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset & Loader Setup
    root_dir = "data/raw/spectralwaste"
    img_size = (320, 320)
    batch_size = 4

    print(f"Loading expanded multimodal dataset from {root_dir}...")
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
    print(f"Total training batches per epoch: {len(train_loader)} (Batch size: {batch_size})")

    # 2. Compute Class Frequencies & Balanced Inverse Weights
    # Background (class 0) is ~78%, so we balance class weights to force the model to learn foreground classes
    class_weights = torch.tensor([0.25, 4.0, 4.0, 1.5, 3.0, 3.5], dtype=torch.float32, device=device)
    print(f"Applied Class Balancing Weights: {class_weights.tolist()}")

    # 3. Initialize Trainer
    trainer = MultimodalTrainer(
        device=device,
        save_dir="models/checkpoints",
        class_weights=class_weights
    )

    # 4. Execute Progressive Training: 3 Epochs Warmup + 7 Epochs Full Fine-Tuning
    print("\n--- Phase 1: Backbone Freeze Warmup (3 Epochs) ---")
    trainer.model.freeze_backbone()
    optimizer_s1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.model.parameters()),
        lr=0.001,
        weight_decay=0.0005
    )
    for epoch in range(3):
        metrics = trainer.train_epoch(train_loader, optimizer_s1)
        print(f"Stage 1 [Epoch {epoch+1}/3] Loss: {metrics['loss_total']:.4f} (CE: {metrics['loss_ce']:.4f}, Dice: {metrics['loss_dice']:.4f})")

    print("\n--- Phase 2: End-to-End Fine-Tuning with Cosine Annealing (7 Epochs) ---")
    trainer.model.unfreeze_all()
    optimizer_s2 = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=0.0001,
        weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=7, eta_min=1e-6)
    for epoch in range(7):
        metrics = trainer.train_epoch(train_loader, optimizer_s2)
        scheduler.step()
        print(f"Stage 2 [Epoch {epoch+1}/7] Loss: {metrics['loss_total']:.4f} (LR: {scheduler.get_last_lr()[0]:.6f})")

    # 5. Save Final Checkpoint
    checkpoint_path = "models/checkpoints/multimodal_4ch_latest.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        "model_state_dict": trainer.model.state_dict(),
        "config": CONFIG.get("training", {})
    }, checkpoint_path)
    print(f"\n[SUCCESS] Model checkpoint saved to: {checkpoint_path}")

    # 6. Quantitative Evaluation Suite Across Dataset
    print("\n" + "=" * 60)
    print("[EVAL] RUNNING QUANTITATIVE BENCHMARK EVALUATION")
    print("=" * 60)

    eval_suite = ModelEvaluationSuite(num_classes=6)
    trainer.model.eval()

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].cpu().numpy()

            outputs = trainer.model(images)
            preds = torch.argmax(outputs["out"], dim=1).cpu().numpy()

            eval_suite.update(preds, targets)

    results = eval_suite.compute_metrics()
    print(f"Pixel Accuracy:     {results['overall_pixel_accuracy'] * 100:.2f}%")
    print(f"Mean IoU (mIoU):    {results['mean_iou'] * 100:.2f}%")
    print(f"Mean Precision:     {results['mean_precision'] * 100:.2f}%")
    print(f"Mean Recall:        {results['mean_recall'] * 100:.2f}%")
    print(f"Mean F1 (Dice):     {results['mean_f1_score'] * 100:.2f}%")
    print("\nPer-Class Breakdown:")
    for cname, m in results["per_class_metrics"].items():
        if m["total_ground_truth_pixels"] > 0:
            print(f"  - {cname:24s} | IoU: {m['iou']*100:5.2f}% | Prec: {m['precision']*100:5.2f}% | Rec: {m['recall']*100:5.2f}% | F1: {m['f1_dice']*100:5.2f}%")

    print("\nGPU Training & Benchmark Update Completed Successfully!")

if __name__ == "__main__":
    run_gpu_training_job()
