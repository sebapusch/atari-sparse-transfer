from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np


class AgentProtocol(abc.ABC):
    """Protocol for RL Agents."""

    @abc.abstractmethod
    def get_action(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Select an action based on observation."""
        pass

    @abc.abstractmethod
    def update(self, batch: Any) -> Dict[str, float]:
        """Update agent parameters based on a batch of data."""
        pass

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return agent state."""
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load agent state."""
        pass
    
    @abc.abstractmethod
    def to(self, device: torch.device) -> AgentProtocol:
        """Move agent to device."""
        pass
