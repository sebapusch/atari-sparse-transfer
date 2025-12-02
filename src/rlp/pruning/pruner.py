from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List

from rlp.pruning.base import PrunerProtocol
from rlp.pruning.scheduler import SparsityScheduler, ConstantScheduler


class BasePruner(PrunerProtocol):
    """
    Base Pruner that manages masks.
    """
    def __init__(self, scheduler: SparsityScheduler) -> None:
        self.scheduler = scheduler
        self.masks: Dict[str, torch.Tensor] = {}
        self.prunable_layers = (nn.Linear, nn.Conv2d)

    def apply(self, model: nn.Module) -> None:
        """Apply masks to the model weights."""
        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers) and name in self.masks:
                module.weight.data.mul_(self.masks[name])

    def _init_masks(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers):
                self.masks[name] = torch.ones_like(module.weight.data)

    def update(self, model: nn.Module, step: int) -> Dict[str, float]:
        """Default update does nothing unless overridden."""
        return {}

    def on_step(self, model: nn.Module, step: int) -> None:
        """Apply masks on every step to ensure sparsity is maintained."""
        self.apply(model)


class GMPPruner(BasePruner):
    """
    Gradual Magnitude Pruning (Zhu & Gupta, 2017).
    """
    def __init__(self, scheduler: SparsityScheduler, update_frequency: int = 1000) -> None:
        super().__init__(scheduler)
        self.update_frequency = update_frequency

    def update(self, model: nn.Module, step: int) -> Dict[str, float]:
        if not self.masks:
            self._init_masks(model)

        if step % self.update_frequency != 0:
            return {}

        target_sparsity = self.scheduler.get_sparsity(step, 0) # total_steps not needed for these schedulers
        
        # Global pruning or Layer-wise? 
        # Usually layer-wise for stability in RL, or global.
        # Let's implement global pruning for simplicity and effectiveness.
        
        all_weights = []
        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers):
                all_weights.append(module.weight.data.abs().flatten())
        
        if not all_weights:
            return {}

        global_weights = torch.cat(all_weights)
        k = int(target_sparsity * global_weights.numel())
        if k == 0:
            return {"sparsity": 0.0}
            
        threshold = torch.kthvalue(global_weights, k).values.item()

        current_sparsity = 0.0
        total_params = 0
        zero_params = 0

        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers):
                mask = (module.weight.data.abs() > threshold).float()
                self.masks[name] = mask
                module.weight.data.mul_(mask)
                
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()

        current_sparsity = zero_params / total_params if total_params > 0 else 0.0
        return {"sparsity": current_sparsity, "threshold": threshold}


class RandomPruner(BasePruner):
    """
    Random Pruning (for baselines).
    """
    def __init__(self, scheduler: SparsityScheduler, update_frequency: int = 1000) -> None:
        super().__init__(scheduler)
        self.update_frequency = update_frequency

    def update(self, model: nn.Module, step: int) -> Dict[str, float]:
        if not self.masks:
            self._init_masks(model)

        if step % self.update_frequency != 0:
            return {}

        target_sparsity = self.scheduler.get_sparsity(step, 0)
        
        total_params = 0
        zero_params = 0

        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers):
                # Random mask
                mask = (torch.rand_like(module.weight.data) > target_sparsity).float()
                self.masks[name] = mask
                module.weight.data.mul_(mask)
                
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()

        current_sparsity = zero_params / total_params if total_params > 0 else 0.0
        return {"sparsity": current_sparsity}
