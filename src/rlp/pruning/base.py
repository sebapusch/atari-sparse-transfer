from __future__ import annotations

import abc
from typing import Any, Dict, Optional
import torch.nn as nn


class PrunerProtocol(abc.ABC):
    """Protocol for Pruners."""

    @abc.abstractmethod
    def apply(self, model: nn.Module) -> None:
        """Apply masks to the model (zero out pruned weights)."""
        pass

    @abc.abstractmethod
    def update(self, model: nn.Module, step: int) -> Dict[str, float]:
        """Update masks based on step/metrics."""
        pass

    @abc.abstractmethod
    def on_step(self, model: nn.Module, step: int) -> None:
        """Callback for every step (e.g. for dynamic sparsity)."""
        pass
