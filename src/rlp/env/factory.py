from __future__ import annotations

import gymnasium as gym
import numpy as np
import ale_py
from typing import Callable, Optional

# Assuming we might need wrappers. 
# For now, standard Gymnasium wrappers.
# If MinAtar is used, it might need specific handling.

def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    gamma: float = 0.99,
) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Atari specific wrappers
        if "NoFrameskip" in env_id:
             env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=True, grayscale_obs=True, scale_obs=False)
             env = gym.wrappers.FrameStackObservation(env, 4)
        
        # MinAtar specific wrappers would go here
        
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        # Safety check for memory usage
        if np.prod(env.observation_space.shape) > 100000: # > 100KB per frame
             import warnings
             warnings.warn(
                 f"Large observation shape {env.observation_space.shape} detected. "
                 "If using Atari, ensure you use 'NoFrameskip-v4' versions (e.g., PongNoFrameskip-v4) "
                 "to trigger preprocessing wrappers. Otherwise, ReplayBuffer may OOM."
             )
             
        return env

    return thunk
