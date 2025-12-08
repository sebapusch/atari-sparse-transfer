from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, Optional, List, Tuple

from rlp.pruning.base import PrunerProtocol
from rlp.pruning.scheduler import SparsityScheduler
from rlp.pruning.utils import get_prunable_modules, calculate_sparsity


class BasePruner(PrunerProtocol):
    """
    Base Pruner using torch.nn.utils.prune.
    """
    def __init__(self, scheduler: SparsityScheduler | None = None) -> None:
        self.scheduler = scheduler
        self.prunable_types = (nn.Linear, nn.Conv2d)

    def prune(self, model: nn.Module, step: int) -> float | None:
        """Update masks based on step."""
        return None


class GMPPruner(BasePruner):
    """
    Gradual Magnitude Pruning (Zhu & Gupta, 2017) using Global Unstructured Pruning.
    """
    def __init__(self, scheduler: SparsityScheduler, update_frequency: int = 1000) -> None:
        super().__init__(scheduler)
        self.update_frequency = update_frequency

    def prune(self, model: nn.Module, step: int) -> float | None:
        if step % self.update_frequency != 0:
            return None

        target_sparsity = self.scheduler.get_sparsity(step, 0)
        
        # To achieve target global sparsity S:
        # If we use `amount=S` (float), it prunes S fraction of *current* weights.
        # If we want total sparsity S_total, and current is S_curr:
        # We need to prune (S_total - S_curr) / (1 - S_curr) of remaining weights.
        
        current_sparsity = calculate_sparsity(model)
        if target_sparsity <= current_sparsity:
            return None

        # Calculate amount to prune from remaining
        if current_sparsity >= 1.0:
            amount_to_prune = 0.0
        else:
            amount_to_prune = (target_sparsity - current_sparsity) / (1.0 - current_sparsity)
            
        # Clamp for safety
        amount_to_prune = max(0.0, min(1.0, amount_to_prune))

        if amount_to_prune <= 0.0:
            return None

        parameters_to_prune = get_prunable_modules(model)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount_to_prune,
        )
            
        return calculate_sparsity(model)


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
