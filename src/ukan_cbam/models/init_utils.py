"""Initialization utilities for UKAN-CBAM."""

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    """Initialize model weights."""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def get_model_info(model):
    """Get model information."""
    return {
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "flops": 0,  # Will be calculated during inference
        "model_size": 0  # Will be calculated during inference
    }
