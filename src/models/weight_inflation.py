import torch
import torch.nn as nn
from typing import Optional

def inflate_first_conv_weights(
    old_conv: nn.Conv2d,
    target_in_channels: int = 4,
    mode: str = "mean_rgb"
) -> nn.Conv2d:
    """
    Expands the input channels of a 2D Convolutional layer from 3 (RGB) to 4 (RGB + NIR/SWIR)
    without destroying pretrained COCO/ImageNet spatial feature representations.
    
    Formula from Technical Architecture:
    W_conv1^(4)[:, 0:3, :, :] = W_RGB
    W_conv1^(4)[:, 3, :, :]   = (1/3) * sum_{c=0}^2 W_RGB[:, c, :, :] (or zeros)
    
    Args:
        old_conv: Original 3-channel Conv2d layer from pretrained backbone.
        target_in_channels: Desired input channels (default 4).
        mode: Method to initialize the 4th channel ('mean_rgb' or 'zeros').
        
    Returns:
        new_conv: Initialized Conv2d layer accepting `target_in_channels`.
    """
    if old_conv.in_channels == target_in_channels:
        return old_conv

    w_old = old_conv.weight.data  # [out_channels, 3, k_h, k_w]
    out_channels, in_channels, k_h, k_w = w_old.shape

    new_conv = nn.Conv2d(
        in_channels=target_in_channels,
        out_channels=out_channels,
        kernel_size=(k_h, k_w),
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=w_old.device,
        dtype=w_old.dtype
    )

    with torch.no_grad():
        # Preserve original RGB weights
        new_conv.weight.data[:, 0:in_channels, :, :] = w_old

        # Initialize the 4th (NIR/SWIR) channel
        if mode == "mean_rgb":
            # Average across RGB channels to preserve activation magnitude
            w_mean = w_old.mean(dim=1, keepdim=True)
            new_conv.weight.data[:, in_channels:target_in_channels, :, :] = w_mean
        elif mode == "zeros":
            new_conv.weight.data[:, in_channels:target_in_channels, :, :] = 0.0
        else:
            raise ValueError(f"Unsupported inflation mode: {mode}")

        # Copy bias if present
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    return new_conv


def build_4channel_model(
    base_model_name: str = "yolov8m-seg.pt",
    num_classes: int = 6,
    pretrained: bool = True
) -> nn.Module:
    """
    Constructs a 4-channel segmentation model by loading a pretrained YOLO-seg
    backbone and inflating the primary input layer.
    """
    from ultralytics import YOLO

    yolo = YOLO(base_model_name)
    pyt_model = yolo.model

    # Locate and inflate the first convolutional layer
    # In YOLOv8, the first layer is model[0].conv
    first_conv = pyt_model.model[0].conv
    inflated_conv = inflate_first_conv_weights(first_conv, target_in_channels=4, mode="mean_rgb")
    pyt_model.model[0].conv = inflated_conv

    return pyt_model
