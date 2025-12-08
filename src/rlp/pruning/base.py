from __future__ import annotations

import abc
from typing import Any, Dict, Optional
import torch.nn as nn


class PrunerProtocol(abc.ABC):
    """Protocol for Pruners."""

    @abc.abstractmethod
    def prune(self, model: nn.Module, step: int) -> float | None:
        """
        Prune the model. Returns the sparsity level. **Returns None values
        if no pruning was applied.**
        """
        pass