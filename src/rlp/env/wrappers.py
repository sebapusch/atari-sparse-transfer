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
