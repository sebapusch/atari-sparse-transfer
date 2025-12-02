from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional
import os

def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    return torch.load(path, map_location=device)

def extract_mask(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract binary masks from a model (where weights are 0)."""
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            masks[name] = (param.data != 0).float()
    return masks

def apply_mask_to_model(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Apply masks to a model."""
    for name, param in model.named_parameters():
        if name in masks:
            # Ensure mask is on same device
            mask = masks[name].to(param.device)
            # Resize if necessary (simple interpolation for now, or error)
            if mask.shape != param.shape:
                # TODO: Implement smart resizing if needed
                print(f"Warning: Mask shape {mask.shape} mismatch with param {param.shape} for {name}. Skipping.")
                continue
            
            param.data.mul_(mask)

def reset_weights_under_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Re-initialize weights but keep the mask structure."""
    # This usually means re-init everything, then apply mask.
    # Or re-init only zeroed weights?
    # Usually we re-init everything using standard init, then zero out pruned.
    
    # Re-init model (assuming standard init is sufficient or custom init called before)
    # Here we just apply the mask again to ensure zeros are zeros.
    apply_mask_to_model(model, masks)
