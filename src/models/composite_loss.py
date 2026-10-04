import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class DiceLoss(nn.Module):
    """
    Dice Loss for binary and multi-class segmentation mask optimization:
    L_Dice = 1 - (2 * sum(p_i * y_i) + eps) / (sum(p_i) + sum(y_i) + eps)
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits [B, C, H, W]
            targets: Ground-truth target indices [B, H, W]
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Flatten spatial dimensions
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=-1)
        cardinality = probs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        return dice_loss


class CompositeSegmentationLoss(nn.Module):
    """
    Formulates the composite loss balancing multi-class cross-entropy and Dice loss:
    L_total = lambda_cls * L_CE + lambda_mask * L_Dice
    """

    def __init__(
        self,
        num_classes: int = 6,
        lambda_cls: float = 0.5,
        lambda_mask: float = 2.5,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_mask = lambda_mask
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_ce = self.ce_loss(logits, targets)
        l_dice = self.dice_loss(logits, targets)
        l_total = self.lambda_cls * l_ce + self.lambda_mask * l_dice

        return {
            "loss_total": l_total,
            "loss_ce": l_ce,
            "loss_dice": l_dice
        }
