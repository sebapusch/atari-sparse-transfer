from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    """Base class for Encoders."""
    
    @property
    def output_dim(self) -> int:
        raise NotImplementedError


class NatureCNN(Encoder):
    """
    DQN Nature Paper CNN.
    Input: (B, C, H, W) usually (B, 4, 84, 84)
    """

    def __init__(self, input_channels: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate output dimension
        # 84x84 -> 20x20 -> 9x9 -> 7x7
        # 64 * 7 * 7 = 3136
        self._output_dim = 3136

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expecting float input, usually normalized / 255.0 externally or here
        return self.network(x)


class MinAtarCNN(Encoder):
    """
    CNN for MinAtar environments.
    Input: (B, C, 10, 10)
    """

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 10x10 -> 8x8
        # 16 * 8 * 8 = 1024
        self._output_dim = 1024

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
