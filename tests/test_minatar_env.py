import gymnasium as gym
import numpy as np
from rlp.env.factory import make_env

def test_minatar_env_creation():
    env_id = "MinAtar/breakout"
    env_fn = make_env(env_id, seed=1, idx=0, capture_video=False, run_name="test")
    env = env_fn()
    
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    
    # Check observation shape: (C, H, W) -> (4, 10, 10) for Breakout
    obs, _ = env.reset()
    assert obs.shape == (4, 10, 10)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    
    # Step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (4, 10, 10)
    assert isinstance(reward, (float, np.floating, int, np.integer))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    env.close()

if __name__ == "__main__":
    try:
        test_minatar_env_creation()
        print("MinAtar environment test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
