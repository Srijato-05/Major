import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from typing import Dict, Optional
from src.models.weight_inflation import inflate_first_conv_weights

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    smp = None
    HAS_SMP = False


class Multimodal4ChSegmentationModel(nn.Module):
    """
    End-to-End 4-Channel Multi-Modal Deep Neural Network for E-Waste Segmentation.
    Defaults to DeepLabV3+ with ResNet-34 encoder (stride-4 skip connections for sharp
    copper wire & IC chip edges, ~3x faster on GTX 1650), with fallback to torchvision DeepLabV3-ResNet50.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 4,
        pretrained: bool = True,
        architecture: Optional[str] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        if architecture is None:
            from src.utils.config_loader import CONFIG
            architecture = CONFIG.get("model", {}).get("architecture", "deeplabv3plus_resnet34")
        self.architecture = architecture

        if HAS_SMP and self.architecture == "fpn_mit_b2":
            encoder_name = "mit_b2"
            self.backend = "smp"
            try:
                self.base_model = smp.FPN(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=in_channels,
                    classes=num_classes,
                )
            except Exception as e:
                print(f"Warning: Could not load pretrained {encoder_name} weights ({e}); initializing randomly.")
                self.base_model = smp.FPN(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=num_classes,
                )
        elif HAS_SMP and "deeplabv3plus" in self.architecture:
            encoder_name = "resnet50" if "resnet50" in self.architecture else "resnet34"
            self.backend = "smp"
            try:
                self.base_model = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=in_channels,
                    classes=num_classes,
                )
            except Exception as e:
                print(f"Warning: Could not load pretrained {encoder_name} weights ({e}); initializing randomly.")
                self.base_model = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=num_classes,
                )
        else:
            self.backend = "torchvision"
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            self.base_model = deeplabv3_resnet50(weights=weights)

            old_conv1 = self.base_model.backbone.conv1
            self.base_model.backbone.conv1 = inflate_first_conv_weights(
                old_conv1,
                target_in_channels=in_channels,
                mode="mean_rgb"
            )

            last_conv = self.base_model.classifier[4]
            self.base_model.classifier[4] = nn.Conv2d(
                in_channels=last_conv.in_channels,
                out_channels=num_classes,
                kernel_size=last_conv.kernel_size,
                stride=last_conv.stride
            )

            if self.base_model.aux_classifier is not None:
                aux_last = self.base_model.aux_classifier[4]
                self.base_model.aux_classifier[4] = nn.Conv2d(
                    in_channels=aux_last.in_channels,
                    out_channels=num_classes,
                    kernel_size=aux_last.kernel_size,
                    stride=aux_last.stride
                )

    def freeze_backbone(self):
        """Freezes backbone/encoder layers except the first 4-channel input layer."""
        if self.backend == "smp":
            for name, param in self.base_model.encoder.named_parameters():
                if "conv1" not in name and "patch_embed1" not in name:
                    param.requires_grad = False
        else:
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
        if self.backend == "smp":
            return {"out": self.base_model(x)}
        return self.base_model(x)


def build_4channel_segmentation_model(
    num_classes: Optional[int] = None,
    in_channels: Optional[int] = None,
    pretrained: bool = True,
    architecture: Optional[str] = None
) -> Multimodal4ChSegmentationModel:
    """Builds the 4-channel segmentation model using system configuration defaults."""
    from src.utils.config_loader import CONFIG

    cfg_model = CONFIG.get("model", {})
    if num_classes is None:
        num_classes = cfg_model.get("num_classes", 7)
    if in_channels is None:
        in_channels = cfg_model.get("in_channels", 4)
    if architecture is None:
        architecture = cfg_model.get("architecture", "deeplabv3plus_resnet34")

    return Multimodal4ChSegmentationModel(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        architecture=architecture
    )
