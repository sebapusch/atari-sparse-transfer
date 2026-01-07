import gymnasium as gym
import numpy as np
from typing import Optional

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.fire_action = -1
        
        # Determine if FIRE is needed
        meanings = env.unwrapped.get_action_meanings()
        if 'FIRE' in meanings:
            self.fire_action = meanings.index('FIRE')

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        # Check for life loss
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # We lost a life, but the game is not over
            terminated = True
            
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step OR fire to advance from terminal/lost life state
            action = 0
            if self.fire_action != -1:
                action = self.fire_action
                
            obs, _, _, _, info = self.env.step(action)
            
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.fire_action = env.unwrapped.get_action_meanings().index('FIRE')

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(self.fire_action)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2) # Step 2 is often 'UP' or 'RIGHT' to break symmetry, keeping consistent with common baselines
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}

    def step(self, ac):
        return self.env.step(ac)



class MinAtarToGymWrapper(gym.Env):
    """
    Wrapper to make MinAtar environments compatible with Gymnasium.
    MinAtar envs return a boolean grid of shape (10, 10, C).
    We convert this to a float32 box of shape (C, 10, 10) for PyTorch.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_name: str):
        from minatar import Environment
        self.env_name = env_name
        self.env = Environment(env_name)
        
        # MinAtar state is (10, 10, C) boolean
        h, w, c = self.env.state_shape()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )
        
        self.action_space = gym.spaces.Discrete(self.env.num_actions())
        self.render_mode = "rgb_array"

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.env.reset()
        # Seed handling for MinAtar is not directly exposed in the same way as Gym
        # But we can rely on global seeding if needed, or just proceed.
        if seed is not None:
            # MinAtar uses numpy random, so we can seed numpy if we really want to force it
            # But usually it's better to let the global seeder handle it if possible
            # or re-instantiate. For now, we just reset.
            pass
            
        return self._get_obs(), {}

    def step(self, action):
        reward, terminated = self.env.act(action)
        obs = self._get_obs()
        truncated = False # MinAtar doesn't have truncation by default
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # (10, 10, C) -> (C, 10, 10) and float
        state = self.env.state()
        return np.transpose(state, (2, 0, 1)).astype(np.float32)

    def render(self):
        return self.env.display_state()

    def close(self):
        pass


class MinAtarTransferWrapper(gym.ObservationWrapper):
    """
    Gymnasium Wrapper that intercepts the observation from a 'Target' environment
    and re-maps its channels to match the 'Source' environment's semantic expectations.

    Used for transferring pre-trained encoders (Source) to new environments (Target).

    Input Shape: (C_in, 10, 10)
    Output Shape: (C_out, 10, 10)
    """

    # Presets for common transfers
    # Keys: [Source][Target] -> mapping_list
    PRESETS = {
        'breakout': {
            'space_invaders': [0, 5, -1, 1],
            'seaquest': [0, 4, 3, 5],
        },
        'space_invaders': {
            'breakout': [0, 3, 3, 3, -1, 1],
            'seaquest': [0, 5, 5, 5, 2, 4],
        },
        'seaquest': {
            'breakout': [0, 0, -1, 2, 1, 3, 3, -1, -1, -1],
            'space_invaders': [0, 0, 4, -1, 5, 1, 1, -1, -1, -1],
        }
    }

    def __init__(self, env: gym.Env, mapping_list: list[int]):
        super().__init__(env)
        self.mapping_list = mapping_list

        # Assume input observation space is Box(C_in, H, W)
        # We need to update to Box(C_out, H, W)
        # where C_out = len(mapping_list)

        orig_space = env.observation_space
        if not isinstance(orig_space, gym.spaces.Box):
            raise ValueError("MinAtarTransferWrapper expects a Box observation space.")

        if len(orig_space.shape) != 3:
            raise ValueError(f"Expected 3D observation space (C, H, W), got {orig_space.shape}")

        c_in, h, w = orig_space.shape
        c_out = len(mapping_list)

        # Update observation space to reflect Source channel count
        self.observation_space = gym.spaces.Box(
            low=orig_space.low.min(),  # Assuming uniform low/high in MinAtar
            high=orig_space.high.max(),
            shape=(c_out, h, w),
            dtype=orig_space.dtype
        )

        # Pre-compute valid indices for efficiency
        # Convert mapping list to array
        self._mapping_array = np.array(mapping_list, dtype=int)
        
        # Mask for channels that map to a valid source channel (not -1)
        self._valid_mask = (self._mapping_array != -1)
        
        # The actual indices to pull from the input observation
        self._valid_indices = self._mapping_array[self._valid_mask]

        # Verify indices are within bounds of input space
        if len(self._valid_indices) > 0:
            if self._valid_indices.max() >= c_in or self._valid_indices.min() < 0:
                raise ValueError(f"Mapping indices out of bounds for input channels {c_in}: {self._valid_indices}")

    def observation(self, obs):
        """
        Remap channels from Target environment (obs) to Source environment (output).
        Handles -1 by filling with zeros.
        """
        # obs is (C_in, H, W)
        # Output (C_out, H, W)

        # Create output buffer of zeros with correct shape and dtype
        new_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        # NumPy indexing to fill valid channels
        # new_obs[valid_mask] selects the channels where mapping is not -1 (shape: N_valid, H, W)
        # obs[valid_indices] selects the corresponding source channels from input (shape: N_valid, H, W)
        if np.any(self._valid_mask):
            new_obs[self._valid_mask] = obs[self._valid_indices]

        return new_obs
