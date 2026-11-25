import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from src.data.multimodal_dataset import build_multimodal_dataloader, ManifestMultimodalDataset
from src.models.train_pipeline import MultimodalTrainer


def compute_inverse_log_weights(manifest_path: str, num_classes: int = 7) -> torch.Tensor:
    """Computes balanced inverse-log class weights from exact training split pixel frequencies."""
    ds = ManifestMultimodalDataset(manifest_path=manifest_path, img_size=(128, 128), is_training=False, spectral_dropout_prob=0.0)
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(ds)):
        m = ds[i]["mask"].numpy().flatten()
        counts += np.bincount(m[(m >= 0) & (m < num_classes)], minlength=num_classes)[:num_classes]

    total = max(int(counts.sum()), 1)
    freqs = counts / float(total)
    raw_fg = np.array([1.0 / np.log(1.02 + f) if f > 0 else 1.0 for f in freqs[1:]], dtype=np.float32)
    fg_weights = np.clip(raw_fg / max(float(raw_fg.min()), 1e-6), 1.0, 6.0)
    weights = np.concatenate([[0.30], fg_weights]).astype(np.float32)

    pct_str = [f"c{c}:{freqs[c]*100:.2f}%" for c in range(num_classes)]
    print(f"Train pixel distribution: {', '.join(pct_str)}")
    print(f"Computed class weights:   {[round(float(w), 2) for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Rigorous 7-class multimodal training with AMP, validation logging, and early stopping.")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="models/checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path for warm-start fine-tuning")
    parser.add_argument("--stage2-lr", type=float, default=None, help="Optional override for Stage 2 max LR")
    parser.add_argument("--no-oversample-rare", action="store_true", help="Disable WeightedRandomSampler rare-class oversampling")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print(f"[RIGOROUS 7-CLASS TRAINING] device={device} img_size={args.img_size} batch={args.batch_size} accum={args.grad_accum}")
    print("=" * 75)

    train_loader = build_multimodal_dataloader(
        split="train",
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        is_training=True,
        num_workers=0,
        oversample_rare=not args.no_oversample_rare,
    )
    val_loader = build_multimodal_dataloader(
        split="val",
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        is_training=False,
        num_workers=0,
    )

    class_weights = compute_inverse_log_weights("data/processed/train.json", num_classes=7)

    trainer = MultimodalTrainer(
        device=device,
        save_dir=args.save_dir,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
        use_amp=not args.no_amp,
        grad_accum_steps=args.grad_accum,
    )

    if args.resume and os.path.exists(args.resume):
        chk = torch.load(args.resume, map_location=trainer.device, weights_only=False)
        sd = chk.get("model_state_dict", chk)
        model_sd = trainer.model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        trainer.model.load_state_dict(filtered, strict=False)
        print(f"Resumed warm-start weights ({len(filtered)}/{len(model_sd)} tensors) from: {args.resume}")

    if args.stage2_lr is not None:
        trainer.cfg_train = dict(trainer.cfg_train)
        trainer.cfg_train["stage2_lr_max"] = args.stage2_lr

    results = trainer.run_training_pipeline(
        dataloader=train_loader,
        val_loader=val_loader,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        patience=args.patience,
        max_batches=args.max_batches,
    )
    print(f"Training finished. Best Val mIoU: {results['best_val_miou']*100:.2f}%")


if __name__ == "__main__":
    main()
