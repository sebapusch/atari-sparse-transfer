from __future__ import annotations

import torch
import torch.nn as nn


class Head(nn.Module):
    """Base class for Heads."""
    pass


class LinearHead(Head):
    """Standard DQN Head."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DuelingHead(Head):
    """Dueling DQN Head."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        advantage = self.advantage_stream(x)
        value = self.value_stream(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
