
try:
    from minatar import Environment
    env = Environment('breakout')
    print(f"Environment created: {env}")
    print(f"State shape: {env.state().shape}")
    print(f"Num channels: {env.state_shape()[2]}")
    print(f"Minimal actions: {env.minimal_action_set()}")
except ImportError:
    print("MinAtar not installed")
except Exception as e:
    print(f"Error: {e}")
