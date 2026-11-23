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

        # Aggregate across batch & spatial dimensions so Dice is evaluated per present class
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=(0, 2))
        probs_sum = probs_flat.sum(dim=(0, 2))
        targets_sum = targets_flat.sum(dim=(0, 2))
        cardinality = probs_sum + targets_sum

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        present = targets_sum > 0
        if present.any():
            dice_loss = 1.0 - dice_score[present].mean()
        else:
            dice_loss = 1.0 - dice_score.mean()
        return dice_loss


class CompositeSegmentationLoss(nn.Module):
    """
    Formulates the composite loss balancing multi-class cross-entropy (or focal CE) and Dice loss:
    L_total = lambda_cls * L_CE/Focal + lambda_mask * L_Dice
    """

    def __init__(
        self,
        num_classes: int = 7,
        lambda_cls: float = 0.5,
        lambda_mask: float = 2.5,
        class_weights: torch.Tensor = None,
        focal_gamma: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_mask = lambda_mask
        self.focal_gamma = focal_gamma
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.focal_gamma > 0.0:
            ce_unweighted = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce_unweighted)
            focal_term = ((1.0 - pt) ** self.focal_gamma) * ce_unweighted
            if self.class_weights is not None:
                weights = self.class_weights.to(logits.device)[targets.clamp(0, self.num_classes - 1)]
                l_ce = (focal_term * weights).mean()
            else:
                l_ce = focal_term.mean()
        else:
            l_ce = self.ce_loss(logits, targets)

        l_dice = self.dice_loss(logits, targets)
        l_total = self.lambda_cls * l_ce + self.lambda_mask * l_dice

        return {
            "loss_total": l_total,
            "loss_ce": l_ce,
            "loss_dice": l_dice
        }
