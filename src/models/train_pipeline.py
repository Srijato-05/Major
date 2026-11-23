import os
import csv
import json
import shutil
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any, Optional, List
from src.utils.config_loader import CONFIG
from src.data.multimodal_dataset import build_multimodal_dataloader
from src.models.multimodal_segmenter import build_4channel_segmentation_model, Multimodal4ChSegmentationModel
from src.models.composite_loss import CompositeSegmentationLoss
from src.models.evaluation_suite import ModelEvaluationSuite

class MultimodalTrainer:
    """
    Two-Stage Training & Fine-Tuning Pipeline for 4-Channel E-Waste Segmentation:
    - Stage 1: Feature Extraction Freeze (frozen backbone, training inflated Conv1 and heads with AdamW)
    - Stage 2: End-to-End Unfreezing with Cosine Annealing Learning Rate Schedule
    Supports PyTorch AMP (FP16), gradient accumulation, Focal+Dice loss, validation tracking, and early stopping.
    """

    def __init__(
        self,
        model: Optional[Multimodal4ChSegmentationModel] = None,
        device: Optional[str] = None,
        save_dir: str = "models/checkpoints",
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 0.0,
        use_amp: bool = True,
        grad_accum_steps: int = 1
    ):
        cfg_train = CONFIG.get("training", {})

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.num_classes = CONFIG.get("model", {}).get("num_classes", 7)
        self.model = model or build_4channel_segmentation_model(num_classes=self.num_classes, pretrained=True)
        self.model.to(self.device)

        self.use_amp = bool(use_amp and self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        loss_weights = cfg_train.get("loss_weights", {})
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = CompositeSegmentationLoss(
            num_classes=self.num_classes,
            lambda_cls=loss_weights.get("lambda_cls", 0.5),
            lambda_mask=loss_weights.get("lambda_mask", 2.5),
            class_weights=class_weights,
            focal_gamma=focal_gamma
        )

        self.cfg_train = cfg_train

    def train_epoch(self, dataloader, optimizer, max_batches: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        num_batches = 0
        optimizer.zero_grad()

        iterator = tqdm(dataloader, desc="Train", leave=False) if hasattr(dataloader, "__len__") else dataloader
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(self.device)
            targets = batch["mask"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                logits = outputs["out"]
                loss_dict = self.criterion(logits, targets)
                raw_loss = loss_dict["loss_total"]
                if "aux" in outputs and outputs["aux"] is not None:
                    aux_loss_dict = self.criterion(outputs["aux"], targets)
                    raw_loss = raw_loss + 0.4 * aux_loss_dict["loss_total"]
                loss = raw_loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            total_loss += loss_dict["loss_total"].item()
            total_ce += loss_dict["loss_ce"].item()
            total_dice += loss_dict["loss_dice"].item()
            num_batches += 1

        if num_batches > 0 and num_batches % self.grad_accum_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()

        return {
            "loss_total": total_loss / max(num_batches, 1),
            "loss_ce": total_ce / max(num_batches, 1),
            "loss_dice": total_dice / max(num_batches, 1)
        }

    @torch.no_grad()
    def evaluate_epoch(self, dataloader, max_batches: Optional[int] = None) -> Dict[str, Any]:
        """Evaluates loss and segmentation metrics across a validation or test DataLoader."""
        self.model.eval()
        evaluator = ModelEvaluationSuite(num_classes=self.num_classes)
        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        num_batches = 0

        iterator = tqdm(dataloader, desc="Eval", leave=False) if hasattr(dataloader, "__len__") else dataloader
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch["image"].to(self.device)
            targets = batch["mask"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                logits = outputs["out"]
                loss_dict = self.criterion(logits, targets)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            evaluator.update(preds, targets.cpu().numpy())

            total_loss += loss_dict["loss_total"].item()
            total_ce += loss_dict["loss_ce"].item()
            total_dice += loss_dict["loss_dice"].item()
            num_batches += 1

        metrics = evaluator.compute_metrics()
        metrics["loss_total"] = total_loss / max(num_batches, 1)
        metrics["loss_ce"] = total_ce / max(num_batches, 1)
        metrics["loss_dice"] = total_dice / max(num_batches, 1)
        return metrics

    def _save_history(self, history: Dict[str, Any]) -> None:
        json_path = os.path.join(self.save_dir, "training_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        csv_path = os.path.join(self.save_dir, "training_history.csv")
        rows: List[Dict[str, Any]] = history.get("epochs", [])
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

    def run_training_pipeline(
        self,
        dataloader,
        val_loader=None,
        stage1_epochs: Optional[int] = None,
        stage2_epochs: Optional[int] = None,
        patience: int = 5,
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes the progressive two-stage fine-tuning pipeline with validation monitoring,
        best-checkpoint saving, early stopping, and epoch history export.
        """
        s1_epochs = stage1_epochs if stage1_epochs is not None else self.cfg_train.get("stage1_freeze_epochs", 2)
        s2_epochs = stage2_epochs if stage2_epochs is not None else self.cfg_train.get("stage2_unfreeze_epochs", 5)

        history: Dict[str, Any] = {"stage1": [], "stage2": [], "epochs": []}
        best_val_miou = -1.0
        best_checkpoint_path = os.path.join(self.save_dir, "multimodal_4ch_best.pth")
        latest_checkpoint_path = os.path.join(self.save_dir, "multimodal_4ch_latest.pth")

        def _record_epoch(stage: str, epoch_idx: int, train_metrics: Dict[str, float], lr: float, allow_early_stop: bool = False):
            nonlocal best_val_miou
            val_metrics = self.evaluate_epoch(val_loader, max_batches=max_batches) if val_loader is not None else None
            row = {
                "stage": stage,
                "epoch": epoch_idx,
                "lr": lr,
                "train_loss": round(train_metrics["loss_total"], 6),
                "train_ce": round(train_metrics["loss_ce"], 6),
                "train_dice": round(train_metrics["loss_dice"], 6),
                "val_loss": round(val_metrics["loss_total"], 6) if val_metrics else None,
                "val_miou": round(val_metrics["mean_iou"], 4) if val_metrics else None,
                "val_mf1": round(val_metrics["mean_f1_score"], 4) if val_metrics else None,
                "val_pixel_acc": round(val_metrics["overall_pixel_accuracy"], 4) if val_metrics else None,
            }
            history["epochs"].append(row)

            improved = False
            if val_metrics is not None:
                history_entry = dict(train_metrics)
                history_entry["val_loss"] = val_metrics["loss_total"]
                history_entry["val_miou"] = val_metrics["mean_iou"]
                history_entry["val_mf1"] = val_metrics["mean_f1_score"]
                history_entry["per_class_val"] = val_metrics["per_class_metrics"]
                history[stage].append(history_entry)

                if val_metrics["mean_iou"] > best_val_miou:
                    best_val_miou = val_metrics["mean_iou"]
                    improved = True
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "config": self.cfg_train,
                        "best_val_miou": best_val_miou,
                        "epoch": epoch_idx,
                        "stage": stage
                    }, best_checkpoint_path)
            else:
                history[stage].append(train_metrics)

            self._save_history(history)
            return val_metrics, improved

        # ---------------- STAGE 1: BACKBONE FREEZE ----------------
        print(f"=== Starting Stage 1: Backbone Freeze ({s1_epochs} epochs) ===")
        self.model.freeze_backbone()
        s1_lr = self.cfg_train.get("stage1_lr", 0.001)
        optimizer_s1 = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=s1_lr,
            weight_decay=self.cfg_train.get("weight_decay", 0.0005)
        )

        for epoch in range(s1_epochs):
            metrics = self.train_epoch(dataloader, optimizer_s1, max_batches=max_batches)
            val_m, _ = _record_epoch("stage1", epoch + 1, metrics, s1_lr)
            val_str = f" | Val Loss: {val_m['loss_total']:.4f}, Val mIoU: {val_m['mean_iou']*100:.2f}%" if val_m else ""
            print(f"Stage 1 [Epoch {epoch+1}/{s1_epochs}] Loss: {metrics['loss_total']:.4f} (CE: {metrics['loss_ce']:.4f}, Dice: {metrics['loss_dice']:.4f}){val_str}")

        # ---------------- STAGE 2: FULL UNFREEZE ----------------
        print(f"\n=== Starting Stage 2: End-to-End Fine-Tuning ({s2_epochs} epochs) ===")
        self.model.unfreeze_all()
        lr_max = self.cfg_train.get("stage2_lr_max", 0.0001)
        lr_min = self.cfg_train.get("stage2_lr_min", 0.000001)
        optimizer_s2 = optim.AdamW(
            self.model.parameters(),
            lr=lr_max,
            weight_decay=self.cfg_train.get("weight_decay", 0.0005)
        )
        scheduler = CosineAnnealingLR(optimizer_s2, T_max=max(1, s2_epochs), eta_min=lr_min)

        no_improve_epochs = 0
        for epoch in range(s2_epochs):
            metrics = self.train_epoch(dataloader, optimizer_s2, max_batches=max_batches)
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0]
            val_m, improved = _record_epoch("stage2", epoch + 1, metrics, curr_lr, allow_early_stop=True)
            val_str = f" | Val Loss: {val_m['loss_total']:.4f}, Val mIoU: {val_m['mean_iou']*100:.2f}%" if val_m else ""
            print(f"Stage 2 [Epoch {epoch+1}/{s2_epochs}] Loss: {metrics['loss_total']:.4f} (LR: {curr_lr:.6f}){val_str}")

            if val_m is not None:
                if improved:
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if patience > 0 and no_improve_epochs >= patience:
                        print(f"Early stopping triggered after {patience} epochs without Val mIoU improvement.")
                        break

        # Save latest checkpoint (copy best validation checkpoint if available so latest == best)
        if val_loader is not None and os.path.exists(best_checkpoint_path):
            shutil.copy2(best_checkpoint_path, latest_checkpoint_path)
            print(f"\nBest validation checkpoint (mIoU={best_val_miou*100:.2f}%) saved to: {best_checkpoint_path} and synced to {latest_checkpoint_path}")
        else:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.cfg_train,
                "best_val_miou": best_val_miou
            }, latest_checkpoint_path)
            print(f"\nLatest checkpoint saved to: {latest_checkpoint_path}")

        return {
            "checkpoint_path": latest_checkpoint_path,
            "best_checkpoint_path": best_checkpoint_path if os.path.exists(best_checkpoint_path) else latest_checkpoint_path,
            "best_val_miou": best_val_miou,
            "history": history
        }
