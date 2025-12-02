from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, Optional, List, Tuple

from rlp.pruning.base import PrunerProtocol
from rlp.pruning.scheduler import SparsityScheduler

class BasePruner(PrunerProtocol):
    """
    Base Pruner using torch.nn.utils.prune.
    """
    def __init__(self, scheduler: Optional[SparsityScheduler] = None) -> None:
        self.scheduler = scheduler
        self.prunable_types = (nn.Linear, nn.Conv2d)

    def update(self, model: nn.Module, step: int) -> Dict[str, float]:
        """Update masks based on step."""
        return {}

    def apply(self, model: nn.Module) -> None:
        """
        Apply masks to the model. 
        With torch.nn.utils.prune, this is handled automatically via hooks.
        We can use this to make pruning permanent if needed, but for now it's a no-op.
        """
        pass

    def on_step(self, model: nn.Module, step: int) -> None:
        """Called every step. Not needed for torch.prune as masks are persistent."""
        pass

    def get_prunable_modules(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
        modules = []
        for name, module in model.named_modules():
            if isinstance(module, self.prunable_types):
                modules.append((module, 'weight'))
        return modules
        
    def calculate_sparsity(self, model: nn.Module) -> float:
        total_params = 0
        zero_params = 0
        for module, name in self.get_prunable_modules(model):
            if prune.is_pruned(module):
                mask = getattr(module, name + "_mask")
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()
            else:
                total_params += getattr(module, name).numel()
        
        return zero_params / total_params if total_params > 0 else 0.0


class GMPPruner(BasePruner):
    """
    Gradual Magnitude Pruning (Zhu & Gupta, 2017) using Global Unstructured Pruning.
    """
    def __init__(self, scheduler: SparsityScheduler, update_frequency: int = 1000) -> None:
        super().__init__(scheduler)
        self.update_frequency = update_frequency

    def update(self, model: nn.Module, step: int) -> Dict[str, float]:
        if step % self.update_frequency != 0:
            return {}

        target_sparsity = self.scheduler.get_sparsity(step, 0)
        
        # GMP is monotonic. We prune to `target_sparsity`.
        # torch.prune.global_unstructured prunes the *currently unpruned* parameters if we call it repeatedly?
        # No, `amount` is fraction of *current* parameters if float, or absolute number if int.
        # But we want target sparsity of the *original* model.
        
        # To achieve target global sparsity S:
        # If we use `amount=S` (float), it prunes S fraction of *current* weights.
        # If we want total sparsity S_total, and current is S_curr:
        # We need to prune (S_total - S_curr) / (1 - S_curr) of remaining weights.
        
        current_sparsity = self.calculate_sparsity(model)
        if target_sparsity <= current_sparsity:
            return {"sparsity": current_sparsity}
            
        # Calculate amount to prune from remaining
        if current_sparsity >= 1.0:
            amount_to_prune = 0.0
        else:
            amount_to_prune = (target_sparsity - current_sparsity) / (1.0 - current_sparsity)
            
        # Clamp for safety
        amount_to_prune = max(0.0, min(1.0, amount_to_prune))
        
        if amount_to_prune > 0:
            parameters_to_prune = self.get_prunable_modules(model)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount_to_prune,
            )
            
        return {"sparsity": self.calculate_sparsity(model)}


class LTHPruner(BasePruner):
    """
    Lottery Ticket Hypothesis / Iterative Magnitude Pruning.
    Prunes p% of weights every k steps.
    """
    def __init__(self, pruning_rate: float = 0.2, pruning_steps: List[int] = None) -> None:
        super().__init__(None) # No continuous scheduler
        self.pruning_rate = pruning_rate
        self.pruning_steps = set(pruning_steps) if pruning_steps else set()

    def update(self, model: nn.Module, step: int) -> Dict[str, float]:
        if step not in self.pruning_steps:
            return {}
            
        # Prune p% of remaining weights globally
        parameters_to_prune = self.get_prunable_modules(model)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.pruning_rate,
        )
        
        return {"sparsity": self.calculate_sparsity(model)}


class RandomPruner(BasePruner):
    """
    Random Pruning using torch.prune.
    """
    def __init__(self, scheduler: SparsityScheduler, update_frequency: int = 1000) -> None:
        super().__init__(scheduler)
        self.update_frequency = update_frequency

    def update(self, model: nn.Module, step: int) -> Dict[str, float]:
        if step % self.update_frequency != 0:
            return {}

        target_sparsity = self.scheduler.get_sparsity(step, 0)
        current_sparsity = self.calculate_sparsity(model)
        
        if target_sparsity <= current_sparsity:
            return {"sparsity": current_sparsity}
            
        amount_to_prune = (target_sparsity - current_sparsity) / (1.0 - current_sparsity)
        amount_to_prune = max(0.0, min(1.0, amount_to_prune))
        
        if amount_to_prune > 0:
            parameters_to_prune = self.get_prunable_modules(model)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount_to_prune,
            )
            
        return {"sparsity": self.calculate_sparsity(model)}
