import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from typing import Dict, Any, Optional
from src.models.weight_inflation import inflate_first_conv_weights

class Multimodal4ChSegmentationModel(nn.Module):
    """
    End-to-End 4-Channel Multi-Modal Deep Neural Network for E-Waste Segmentation.
    Adapts a high-capacity pretrained vision backbone (ResNet50 DeepLabV3)
    from 3 input channels (RGB) to 4 input channels (RGB + NIR/SWIR) via Conv1 weight inflation.
    Outputs dense multi-class segmentation masks across the 6 industrial WEEE categories.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 4,
        pretrained: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Load pretrained backbone
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.base_model = deeplabv3_resnet50(weights=weights)

        # Inflate the 1st convolutional layer (backbone.conv1) to 4 channels
        old_conv1 = self.base_model.backbone.conv1
        self.base_model.backbone.conv1 = inflate_first_conv_weights(
            old_conv1,
            target_in_channels=in_channels,
            mode="mean_rgb"
        )

        # Replace classification head to match target number of classes
        # DeepLabV3 classifier: [Conv2d, BatchNorm2d, ReLU, Conv2d (256 -> num_classes)]
        last_conv = self.base_model.classifier[4]
        self.base_model.classifier[4] = nn.Conv2d(
            in_channels=last_conv.in_channels,
            out_channels=num_classes,
            kernel_size=last_conv.kernel_size,
            stride=last_conv.stride
        )

        # Also update auxiliary classifier if present
        if self.base_model.aux_classifier is not None:
            aux_last = self.base_model.aux_classifier[4]
            self.base_model.aux_classifier[4] = nn.Conv2d(
                in_channels=aux_last.in_channels,
                out_channels=num_classes,
                kernel_size=aux_last.kernel_size,
                stride=aux_last.stride
            )

    def freeze_backbone(self):
        """Freezes all backbone layers except the newly inflated Conv1 layer."""
        for name, param in self.base_model.backbone.named_parameters():
            if "conv1" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes all layers for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input 4-channel tensor [B, 4, H, W]
        Returns:
            dict containing 'out': logits [B, num_classes, H, W]
        """
        out = self.base_model(x)
        return out


def build_4channel_segmentation_model(
    num_classes: Optional[int] = None,
    in_channels: Optional[int] = None,
    pretrained: bool = True
) -> Multimodal4ChSegmentationModel:
    """Builds the 4-channel segmentation model using system configuration defaults."""
    from src.utils.config_loader import CONFIG

    if num_classes is None:
        num_classes = CONFIG.get("model", {}).get("num_classes", 6)
    if in_channels is None:
        in_channels = CONFIG.get("model", {}).get("in_channels", 4)

    return Multimodal4ChSegmentationModel(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained
    )
