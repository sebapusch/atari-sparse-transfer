from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rlp.agent.base import AgentProtocol
from rlp.components.network import QNetwork


class DQNAgent(AgentProtocol):
    """
    DQN Agent with support for Double DQN.
    """

    def __init__(
        self,
        network: QNetwork,
        optimizer: optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 1.0, # 1.0 = hard update
        target_network_frequency: int = 1000,
        double_dqn: bool = False,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.target_network = copy.deepcopy(network).to(self.device)
        self.optimizer = optimizer
        
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.double_dqn = double_dqn
        
        self.steps = 0

    def get_action(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        obs: (B, ...) or (...)
        """
        # Handle single observation vs batch
        if obs.ndim == 3: # (C, H, W) -> (1, C, H, W)
             obs = np.expand_dims(obs, axis=0)
        
        # Epsilon-greedy
        if random.random() < epsilon:
            # Assuming discrete action space size is known or inferred.
            # Ideally we pass action_space to init, but here we can infer from network output
            # if we assume the head is LinearHead or DuelingHead with last layer output_dim.
            # But `network` is generic. Let's assume we can get it from the network output shape.
            # Ideally, `get_action` should take `action_space` or we store `num_actions`.
            # Let's infer num_actions from the network's last layer if possible, or pass it in init.
            # For now, let's assume we can run a forward pass to get shape, or better, pass num_actions to init.
            # Wait, the reference uses `envs.single_action_space.sample()`.
            # The Agent shouldn't depend on `envs`.
            # We will assume the caller handles random sampling if epsilon check passes?
            # No, `get_action` usually handles it.
            # Let's add `num_actions` to __init__.
            pass

        # We need num_actions. Let's update __init__ to take num_actions or infer it.
        # Actually, for efficiency, we should probably just return the Q-values
        # and let the caller decide? No, `get_action` implies returning an action.
        
        # Let's assume we do the forward pass first.
        with torch.no_grad():
            q_values = self.network(torch.Tensor(obs).to(self.device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
        # If epsilon > 0, we need to overwrite with random actions.
        # But we don't know num_actions easily without passing it.
        # Let's assume the user passes `num_actions` to `__init__`.
        return actions

    # Re-implementing with num_actions
    def __init__(
        self,
        network: QNetwork,
        optimizer: optim.Optimizer,
        num_actions: int,
        gamma: float = 0.99,
        tau: float = 1.0,
        target_network_frequency: int = 1000,
        double_dqn: bool = False,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.target_network = copy.deepcopy(network).to(self.device)
        self.optimizer = optimizer
        self.num_actions = num_actions
        
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.double_dqn = double_dqn
        
        self.steps = 0

    def get_action(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        if obs.ndim == 3:
             obs = np.expand_dims(obs, axis=0)
        
        batch_size = obs.shape[0]
        
        # Random actions
        if random.random() < epsilon:
            return np.random.randint(0, self.num_actions, size=batch_size)
        
        with torch.no_grad():
            q_values = self.network(torch.Tensor(obs).to(self.device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        return actions

    def update(self, batch: Any) -> Dict[str, float]:
        """
        batch: ReplayBuffer sample (observations, actions, rewards, next_observations, dones)
        """
        self.steps += 1
        
        # Unpack batch (assuming CleanRL ReplayBuffer structure or similar namedtuple/dataclass)
        # We'll assume attributes: observations, actions, rewards, next_observations, dones
        obs = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.flatten().to(self.device)
        next_obs = batch.next_observations.to(self.device)
        dones = batch.dones.flatten().to(self.device)

        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: selection with online, evaluation with target
                next_q_values = self.network(next_obs)
                next_actions = torch.argmax(next_q_values, dim=1)
                target_next_q_values = self.target_network(next_obs)
                target_max = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                target_max, _ = self.target_network(next_obs).max(dim=1)
            
            td_target = rewards + self.gamma * target_max * (1 - dones)

        old_val = self.network(obs).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network update
        if self.steps % self.target_network_frequency == 0:
            for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

        return {
            "loss": loss.item(),
            "q_values": old_val.mean().item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.network.load_state_dict(state_dict["network"])
        self.target_network.load_state_dict(state_dict["target_network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.steps = state_dict["steps"]

    def to(self, device: torch.device) -> DQNAgent:
        self.device = device
        self.network.to(device)
        self.target_network.to(device)
        return self
