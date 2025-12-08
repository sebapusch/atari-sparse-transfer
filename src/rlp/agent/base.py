from __future__ import annotations

import abc
from typing import Any
import torch
import numpy as np

from rlp.core.buffer import ReplayBufferSamples
from rlp.pruning.base import PrunerProtocol


class AgentProtocol(abc.ABC):
    """Protocol for RL Agents."""

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Select an action based on observation."""
        ...

    @abc.abstractmethod
    def update(self, batch: ReplayBufferSamples, step: int) -> dict[str, float]:
        """
        Update agent parameters based on a batch of data.
        Return metrics
        """
        ...

    def finished_training(self, step: int) -> None:
        ...

    @abc.abstractmethod
    def prune(self, step: int) -> float | None:
        ...

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return agent state."""
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load agent state."""
        ...
    
    @abc.abstractmethod
    def to(self, device: torch.device) -> AgentProtocol:
        """Move agent to device."""
        ...