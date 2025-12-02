from __future__ import annotations

import numpy as np
import torch
from typing import NamedTuple, Optional

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

class ReplayBuffer:
    """
    Simple Replay Buffer.
    """
    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple,
        action_shape: tuple,
        device: torch.device,
        n_envs: int = 1,
        obs_dtype: np.dtype = np.float32,
        action_dtype: np.dtype = np.int64,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        
        self.observations = np.zeros((buffer_size, n_envs) + obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((buffer_size, n_envs) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((buffer_size, n_envs) + action_shape, dtype=action_dtype)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict],
    ) -> None:
        # Handle "real_next_obs" logic if needed, but usually passed in as next_obs
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
        # Sample from random envs
        env_inds = np.random.randint(0, self.n_envs, size=batch_size)
        
        return ReplayBufferSamples(
            observations=torch.tensor(self.observations[batch_inds, env_inds], device=self.device),
            actions=torch.tensor(self.actions[batch_inds, env_inds], device=self.device),
            next_observations=torch.tensor(self.next_observations[batch_inds, env_inds], device=self.device),
            dones=torch.tensor(self.dones[batch_inds, env_inds], device=self.device),
            rewards=torch.tensor(self.rewards[batch_inds, env_inds], device=self.device),
        )
