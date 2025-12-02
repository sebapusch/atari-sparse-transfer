from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from rlp.components.encoders import Encoder
from rlp.components.heads import Head


class QNetwork(nn.Module):
    """
    Composite Q-Network: Encoder + Head.
    """

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
        # Normalize pixel inputs if they are uint8 (0-255)
        # Assuming input is float or handled by env wrapper, but standard is / 255.0
        # We'll assume the input is already a float tensor, but if it's > 1.0 we might want to normalize.
        # However, usually normalization happens in the Agent or Env wrapper.
        # Let's stick to the reference: "self.network(x / 255.0)"
        # We will do normalization here for safety if it looks like pixels.
        
        if x.dtype == torch.uint8:
             x = x.float() / 255.0
        elif x.max() > 1.0: # Heuristic check
             x = x / 255.0
             
        features = self.encoder(x)
        q_values = self.head(features)
        return q_values
