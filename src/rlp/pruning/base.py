from __future__ import annotations

import abc
from typing import Any, Dict, Optional
import torch.nn as nn


from dataclasses import dataclass, field

@dataclass
class PruningContext:
    step: int
    agent: Any
    trainer: Any = None
    recent_episodic_returns: list[float] = field(default_factory=list)

class PrunerProtocol(abc.ABC):
    """Protocol for Pruners."""

    @abc.abstractmethod
    def prune(self, model: nn.Module, context: PruningContext) -> float | None:
        """
        Prune the model. Returns the sparsity level. **Returns None values
        if no pruning was applied.**
        """
        pass

    def should_stop(self, context: PruningContext) -> bool:
        """
        Determines if training should stop based on pruning state.
        Default implementation returns False.
        """
        return False