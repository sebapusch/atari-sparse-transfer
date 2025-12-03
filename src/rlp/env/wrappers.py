import gymnasium as gym
import numpy as np

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

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
            self.lives = self.env.unwrapped.ale.lives()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
            
        return obs, info
