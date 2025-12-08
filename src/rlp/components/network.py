from __future__ import annotations

import torch
import torch.nn as nn

from rlp.components.encoders import Encoder
from rlp.components.heads import Head


class QNetwork(nn.Module):
    def __init__(self, encoder: Encoder, head: Head) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Observation tensor.
        Returns:
            Q-values for each action.
        """
        features = self.encoder(x)
        q_values = self.head(features)
        return q_values


