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

    def to(self, device: torch.Device) -> ReplayBufferSamples:
        return ReplayBufferSamples(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            next_observations=self.next_observations.to(device),
            dones=self.dones.to(device),
            rewards=self.rewards.to(device),
        )


class ReplayBuffer:
    """
    Replay Buffer with optional memory optimization (Lazy Frames).
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
        optimize_memory_usage: bool = False,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        self.optimize_memory_usage = optimize_memory_usage
        
        if optimize_memory_usage:
            # Assume obs_shape is (C, H, W) or (Stack, H, W)
            # We store only the last frame: (H, W)
            # We assume channel-first.
            self.storage_shape = obs_shape[1:]
            self.stack_size = obs_shape[0]
        else:
            self.storage_shape = obs_shape
            self.stack_size = 1

        self.observations = np.zeros((buffer_size, n_envs) + self.storage_shape, dtype=obs_dtype)
        
        if not optimize_memory_usage:
            self.next_observations = np.zeros((buffer_size, n_envs) + self.storage_shape, dtype=obs_dtype)
        else:
            self.next_observations = None
            
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
        if self.optimize_memory_usage:
            # Store only the last frame
            # obs is (N, C, H, W). We want (N, H, W)
            self.observations[self.pos] = obs[:, -1, ...]
        else:
            self.observations[self.pos] = obs
            self.next_observations[self.pos] = next_obs
            
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env_inds: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.optimize_memory_usage:
            return (
                torch.tensor(self.observations[batch_inds, env_inds], device=self.device),
                torch.tensor(self.next_observations[batch_inds, env_inds], device=self.device)
            )
            
        # Optimized sampling
        # We need to reconstruct the stack for each batch index
        # obs[i] stack: i-3, i-2, i-1, i
        # next_obs[i] stack: i-2, i-1, i, i+1
        
        # Create a tensor of indices (Batch, Stack)
        # For obs:
        obs_indices = np.zeros((len(batch_inds), self.stack_size), dtype=np.int64)
        next_obs_indices = np.zeros((len(batch_inds), self.stack_size), dtype=np.int64)
        
        for k in range(self.stack_size):
            # For obs at t, we want t - (stack_size - 1) + k
            # e.g. stack=4. k=0 -> t-3. k=3 -> t.
            offset = self.stack_size - 1 - k
            obs_indices[:, k] = batch_inds - offset
            next_obs_indices[:, k] = batch_inds - offset + 1
            
        # Handle wrapping and boundaries
        # We must check if we crossed the 'pos' boundary or 'dones'
        
        # Vectorized validity check is hard, let's do it per sample or semi-vectorized
        # Actually, simpler logic:
        # For each sample, get the window.
        # If we hit a done in the window, mask previous frames with the first valid frame (padding).
        
        # Let's retrieve all potential frames first
        # Handle negative indices (wrapping)
        obs_indices = obs_indices % self.buffer_size
        next_obs_indices = next_obs_indices % self.buffer_size
        
        # We need to check dones. 
        # If we want stack for time t. We look at t-1, t-2...
        # If done[t-1] is True, then t is start of episode. t-1 is irrelevant.
        
        # Fetch frames: (Batch, Stack, N_Envs, H, W) -> Select specific envs
        # This is tricky with numpy advanced indexing for mixed dims
        # self.observations: (Size, N_Envs, H, W)
        # We want (Batch, Stack, H, W)
        
        # Flatten env_inds to broadcast
        # obs_frames: (Batch, Stack, H, W)
        obs_frames = self.observations[obs_indices, env_inds[:, None]]
        next_obs_frames = self.observations[next_obs_indices, env_inds[:, None]]
        
        # Check dones to handle episode boundaries
        # We need dones for the window.
        # For obs at t: check dones at t-1, t-2, t-3...
        # Indices for dones check: same as obs_indices but shifted by -1?
        # Done at t-1 means t is fresh.
        
        # Let's iterate over the stack dimension to apply masking
        # We want to mask frames *before* a done.
        
        # For obs:
        # If done[t-1], then frames t-1, t-2... are invalid for t.
        # We replace them with frame t (or 0, but usually frame t for FrameStack).
        # Actually standard FrameStack repeats the first frame.
        
        # We can iterate from end of stack backwards
        # obs_frames shape: (B, S, H, W)
        # S=4. indices: 0, 1, 2, 3 (current)
        
        for i in range(len(batch_inds)):
            # Handle obs
            current_idx = batch_inds[i]
            env_idx = env_inds[i]
            
            # Check for dones in the window (excluding the current step)
            # We look at dones from t-(S-1) to t-1
            # If done at t-k, then all frames <= t-k are invalid?
            # No, if done at t-1, then t is start. t-1 is end of prev.
            # So frames <= t-1 are invalid.
            
            # We check dones at indices: current_idx - 1, current_idx - 2 ...
            # up to current_idx - (stack_size - 1)
            
            # Optimization: Check if any done exists in the range
            # If so, find the most recent one.
            
            for k in range(1, self.stack_size):
                # Check done at t-k
                check_idx = (current_idx - k) % self.buffer_size
                
                # Also check if we crossed 'pos' (buffer wrap boundary)
                # If we crossed pos, it means we are reading old data that was overwritten?
                # Or if buffer is full, pos is the boundary.
                # If current_idx >= pos and check_idx < pos (wrapped), valid?
                # If current_idx < pos and check_idx >= pos (wrapped), valid?
                # The only invalid case is crossing 'pos' when looking back?
                # Actually, simply checking dones is usually enough if episodes are long.
                # But to be safe, if check_idx == self.pos - 1 (most recently written), and current_idx == self.pos (oldest),
                # then we are crossing boundary.
                
                if self.dones[check_idx, env_idx]:
                    # Found a done at t-k.
                    # Frames from index (S-1-k) downwards are invalid (belong to prev episode).
                    # S=4. k=1 (done at t-1). Invalid: t-1, t-2, t-3.
                    # Valid: t.
                    # We replace invalid frames with the first valid frame (t).
                    # Index in stack: S-1 is t. S-1-k is t-k.
                    # So indices 0 to S-1-k are replaced by frame at S-k (start of episode).
                    
                    valid_start_stack_idx = self.stack_size - k
                    # Copy frame at valid_start_stack_idx to all previous
                    start_frame = obs_frames[i, valid_start_stack_idx]
                    obs_frames[i, :valid_start_stack_idx] = start_frame
                    break
            
            # Handle next_obs
            # Next obs is at t+1.
            # Window: t+1, t, t-1...
            # If done at t, then t+1 is start.
            # If done at t-1, then t is start.
            
            next_current_idx = (current_idx + 1) % self.buffer_size
            
            for k in range(1, self.stack_size):
                check_idx = (next_current_idx - k) % self.buffer_size
                if self.dones[check_idx, env_idx]:
                    valid_start_stack_idx = self.stack_size - k
                    start_frame = next_obs_frames[i, valid_start_stack_idx]
                    next_obs_frames[i, :valid_start_stack_idx] = start_frame
                    break

        return (
            torch.tensor(obs_frames, device=self.device),
            torch.tensor(next_obs_frames, device=self.device)
        )

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        # We must ensure we don't sample indices that don't have enough history?
        # Or just rely on padding/dones.
        # But if we are at index 0, and look back, we wrap to end.
        # If full, end is valid.
        # If not full, end is zeros/invalid.
        
        if not self.full:
            # If not full, we can only sample from [stack_size, pos) to be safe?
            # Or just sample from [0, pos) and handle invalid history (zeros).
            # Since we init with zeros, it's fine (just black frames).
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        else:
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            
        env_inds = np.random.randint(0, self.n_envs, size=batch_size)
        
        obs, next_obs = self._get_samples(batch_inds, env_inds)
        
        return ReplayBufferSamples(
            observations=obs,
            actions=torch.tensor(self.actions[batch_inds, env_inds], device=self.device),
            next_observations=next_obs,
            dones=torch.tensor(self.dones[batch_inds, env_inds], device=self.device),
            rewards=torch.tensor(self.rewards[batch_inds, env_inds], device=self.device),
        )

    def reset(self) -> None:
        """Resets the buffer."""
        self.pos = 0
        self.full = False
        
        # We don't necessarily need to zero out the arrays, 
        # as pos/full control access, but for safety/debugging:
        self.observations.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        
        if self.next_observations is not None:
            self.next_observations.fill(0)
