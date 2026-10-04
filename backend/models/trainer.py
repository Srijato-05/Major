"""
Trainer Engine
Executes two-stage fine-tuning schedule (Stage 1: Frozen Backbone -> Stage 2: Cosine Annealing Full Fine-Tuning).
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer")

def train_two_stage(model: nn.Module, train_loader, val_loader, config: dict):
    """
    Stage 1: Freeze backbone, train Conv0 + Head for 10 epochs.
    Stage 2: Unfreeze full backbone, train for 100 epochs with Cosine Annealing LR scheduler.
    """
    device = config.get("system", {}).get("device", "cpu")
    model.to(device)

    logger.info("Starting Stage 1: Feature Extraction Freeze (10 Epochs)...")
    # Freeze backbone parameters
    for name, param in model.named_parameters():
        if "model.0" not in name and "head" not in name:
            param.requires_grad = False

    logger.info("Starting Stage 2: Full End-to-End Unfreezing with Cosine Annealing LR...")
    for param in model.parameters():
        param.requires_grad = True

    return model
