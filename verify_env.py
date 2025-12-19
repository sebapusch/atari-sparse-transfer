
import gymnasium as gym
import unittest
import numpy as np
from rlp.env.factory import make_env
import ale_py

class TestAtariCorrectness(unittest.TestCase):
    def test_wrapper_stack(self):
        env_id = "BreakoutNoFrameskip-v4"
        # make_env returns a thunk
        env_thunk = make_env(env_id, seed=42, idx=0, capture_video=False, run_name="test")
        env = env_thunk()
        
        print("\nWrapper Stack:")
        current = env
        wrappers = []
        while hasattr(current, 'env'):
            wrappers.append(current.__class__.__name__)
            print(f"- {current.__class__.__name__}")
            current = current.env
        print(f"- {current.__class__.__name__}")
        wrappers.append(current.__class__.__name__)
        
        # Check order
        # Expected: RecoredEpisodeStats -> AtariPreprocessing -> FireResetEnv -> EpisodicLife -> TransformReward -> FrameStack
        # Depending on how gym wraps. FrameStack is usually outer?
        # In factory.py:
        # env = AtariPreprocessing(...)
        # env = FireResetEnv(env)
        # env = EpisodicLifeEnv(env)
        # env = TransformReward(env)
        # env = FrameStackObservation(env)
        
        # So outermost is FrameStackObservation.
        self.assertEqual(wrappers[0], 'FrameStackObservation')
        self.assertEqual(wrappers[1], 'TransformReward')
        self.assertEqual(wrappers[2], 'EpisodicLifeEnv')
        self.assertEqual(wrappers[3], 'FireResetEnv')
        self.assertEqual(wrappers[4], 'AtariPreprocessing')
        
    def test_fire_reset(self):
        # Breakout requires FIRE to start
        env_id = "BreakoutNoFrameskip-v4"
        env_thunk = make_env(env_id, seed=123, idx=0, capture_video=False, run_name="test")
        env = env_thunk()
        
        obs, _ = env.reset()
        # With FireResetEnv, the ball should already be spawned?
        # In Breakout, after FIRE, ball spawns. 
        # Check if we can see the ball? Or just check that we didn't die immediately.
        
        # Taking no-ops checking if game is running
        for _ in range(5):
             obs, r, t, tr, _ = env.step(0) # NOOP
             if t or tr:
                 break
        
        env.close()

if __name__ == '__main__':
    unittest.main()
