import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional
from src.utils.config_loader import CONFIG
from src.data.multimodal_dataset import build_multimodal_dataloader
from src.models.multimodal_segmenter import build_4channel_segmentation_model, Multimodal4ChSegmentationModel
from src.models.composite_loss import CompositeSegmentationLoss

class MultimodalTrainer:
    """
    Two-Stage Training & Fine-Tuning Pipeline for 4-Channel E-Waste Segmentation:
    - Stage 1: Feature Extraction Freeze (frozen backbone, training inflated Conv1 and heads with AdamW)
    - Stage 2: End-to-End Unfreezing with Cosine Annealing Learning Rate Schedule
    """

    def __init__(
        self,
        model: Optional[Multimodal4ChSegmentationModel] = None,
        device: Optional[str] = None,
        save_dir: str = "models/checkpoints",
        class_weights: Optional[torch.Tensor] = None
    ):
        cfg_train = CONFIG.get("training", {})
        cfg_sys = CONFIG.get("system", {})

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = model or build_4channel_segmentation_model(pretrained=True)
        self.model.to(self.device)

        loss_weights = cfg_train.get("loss_weights", {})
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = CompositeSegmentationLoss(
            num_classes=CONFIG.get("model", {}).get("num_classes", 6),
            lambda_cls=loss_weights.get("lambda_cls", 0.5),
            lambda_mask=loss_weights.get("lambda_mask", 2.5),
            class_weights=class_weights
        )

        self.cfg_train = cfg_train

    def train_epoch(self, dataloader, optimizer) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            targets = batch["mask"].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            logits = outputs["out"]

            loss_dict = self.criterion(logits, targets)
            loss = loss_dict["loss_total"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += loss_dict["loss_ce"].item()
            total_dice += loss_dict["loss_dice"].item()
            num_batches += 1

        return {
            "loss_total": total_loss / max(num_batches, 1),
            "loss_ce": total_ce / max(num_batches, 1),
            "loss_dice": total_dice / max(num_batches, 1)
        }

    def run_training_pipeline(
        self,
        dataloader,
        stage1_epochs: Optional[int] = None,
        stage2_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes the complete progressive fine-tuning pipeline.
        """
        s1_epochs = stage1_epochs if stage1_epochs is not None else self.cfg_train.get("stage1_freeze_epochs", 2)
        s2_epochs = stage2_epochs if stage2_epochs is not None else self.cfg_train.get("stage2_unfreeze_epochs", 5)

        history = {"stage1": [], "stage2": []}

        # ---------------- STAGE 1: BACKBONE FREEZE ----------------
        print(f"=== Starting Stage 1: Backbone Freeze ({s1_epochs} epochs) ===")
        self.model.freeze_backbone()
        optimizer_s1 = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg_train.get("stage1_lr", 0.001),
            weight_decay=self.cfg_train.get("weight_decay", 0.0005)
        )

        for epoch in range(s1_epochs):
            metrics = self.train_epoch(dataloader, optimizer_s1)
            history["stage1"].append(metrics)
            print(f"Stage 1 [Epoch {epoch+1}/{s1_epochs}] Loss: {metrics['loss_total']:.4f} (CE: {metrics['loss_ce']:.4f}, Dice: {metrics['loss_dice']:.4f})")

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
        scheduler = CosineAnnealingLR(optimizer_s2, T_max=s2_epochs, eta_min=lr_min)

        for epoch in range(s2_epochs):
            metrics = self.train_epoch(dataloader, optimizer_s2)
            scheduler.step()
            history["stage2"].append(metrics)
            print(f"Stage 2 [Epoch {epoch+1}/{s2_epochs}] Loss: {metrics['loss_total']:.4f} (LR: {scheduler.get_last_lr()[0]:.6f})")

        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, "multimodal_4ch_latest.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.cfg_train
        }, checkpoint_path)
        print(f"\nModel checkpoint saved successfully to: {checkpoint_path}")

        return {
            "checkpoint_path": checkpoint_path,
            "history": history
        }
