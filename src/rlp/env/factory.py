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
        # MinAtar specific wrappers
        if env_id.startswith("MinAtar/"):
            # Extract game name, e.g. MinAtar/breakout -> breakout
            game_name = env_id.split("/")[1]
            from rlp.env.wrappers import MinAtarToGymWrapper
            env = MinAtarToGymWrapper(game_name)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        
        else:
            # Atari specific wrappers
            # Check for legacy "NoFrameskip" or modern "ALE/" prefix
            is_atari = "NoFrameskip" in env_id or env_id.startswith("ALE/")
            
            make_kwargs = {}
            if is_atari:
                # We enforce frameskip=1 at the environment level so AtariPreprocessing
                # (which uses frame_skip=4 by default) can handle the skipping.
                # This prevents "double skipping" errors on ALE/ environments.
                make_kwargs["frameskip"] = 1

            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array", **make_kwargs)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id, **make_kwargs)
                
            env = gym.wrappers.RecordEpisodeStatistics(env)

            if is_atari:
                env = gym.wrappers.AtariPreprocessing(
                    env, 
                    noop_max=30, 
                    frame_skip=4, 
                    screen_size=84, 
                    terminal_on_life_loss=False, 
                    grayscale_obs=True, 
                    scale_obs=True
                )
                
                # FireResetEnv (Only for specific games)
                # Check env_id or unwrapped spec to determine if fire is needed.
                # User specified: Breakout, Pong, SpaceInvaders.
                # We can check the string.
                lower_id = env_id.lower()
                if "breakout" in lower_id or "pong" in lower_id or "spaceinvaders" in lower_id:
                     from rlp.env.wrappers import FireResetEnv
                     env = FireResetEnv(env)

                from rlp.env.wrappers import EpisodicLifeEnv
                env = EpisodicLifeEnv(env)
                env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))
                env = gym.wrappers.FrameStackObservation(env, 4)

        
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
