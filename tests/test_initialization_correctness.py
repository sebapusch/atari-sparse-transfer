
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
from rlp.core.builder import Builder
from rlp.pruning.lottery import LotteryPruner
import torch.nn.utils.prune as prune
import numpy as np

class DummyLogger:
    def log(self, *args, **kwargs): pass
    def close(self): pass

class DummySpace:
    def __init__(self, shape=None, n=None, dtype=np.float32):
        self.shape = shape
        self.n = n
        self.dtype = dtype

class DummyEnv:
    def __init__(self):
        self.single_action_space = DummySpace(shape=(), n=2, dtype=np.int64)
        self.single_observation_space = DummySpace(shape=(4, 84, 84), dtype=np.uint8)
        self.action_space = DummySpace(shape=(), n=2, dtype=np.int64)
        self.observation_space = DummySpace(shape=(4, 84, 84), dtype=np.uint8)
        self.num_envs = 1
    def reset(self, **kwargs): return torch.zeros(1, 4, 84, 84), {}
    def step(self, action): return torch.zeros(1, 4, 84, 84), 0, False, False, {}
    def close(self): pass


class TestInitializationCorrectness:
    
    @patch("rlp.core.builder.Checkpointer")
    def test_full_correctness(self, MockCheckpointer):
        """
        Verifies:
        1. Sparsity is preserved (masks created).
        2. Weights match the artifact.
        3. Theta_0 is loaded into LotteryPruner.
        """
        # 1. Setup Config
        config = OmegaConf.create({
            'initial_artifact': 'entity/project/artifact:v0',
            'algorithm': {
                'name': 'dqn',
                'network': {'encoder': 'nature_cnn', 'head': 'linear', 'hidden_dim': 4},
                'optimizer': {'lr': 1e-4, 'eps': 1e-8},
                'gamma': 0.99, 'tau': 1.0, 'target_network_frequency': 1000,
                'learning_starts': 100, 'train_frequency': 4, 'batch_size': 32,
                'epsilon': {'start': 1.0, 'end': 0.1, 'decay_fraction': 0.1}
            },
            'train': {'total_timesteps': 1000, 'buffer_size': 10000, 'checkpoint_interval': 1000, 'log_interval': 100},
            'pruning': {'method': 'lth', 'final_sparsity': 0.5}, # LTH Pruning enabled
            'env': {'id': 'PongNoFrameskip-v4', 'num_envs': 1, 'capture_video': False, 'frame_stack': 4},
            'seed': 42,
            'wandb': {'enabled': False}, 
            'output_dir': '/tmp/test_output'
        })
        
        # 2. Create Builder
        builder = Builder(config)
        
        # 3. Create Dummy Artifact State
        # We need a REAL state dict structure that matches NatureCNN + LinearHead
        # But we can't easily instantiate them without gym env (for input channels).
        # builder.build_agent needs num_actions, input_channels.
        
        # Let's manually instantiate the network structure match
        # NatureCNN: network (Sequential) -> 0, 2, 4 (Conv), 7 (Linear)
        # LinearHead: network (Linear)
        
        # Let's assume we are targeting head.network.weight for pruning.
        
        # Mask: 50% sparsity (0s and 1s)
        mask = torch.tensor([[1., 0., 1., 0.], [1., 0., 1., 0.]]).detach() # 8 elements, 4 zeros
        orig = torch.randn(2, 4).detach()
        
        # Calculated weight = orig * mask
        weight_pruned = (orig * mask).detach()
        
        # Theta_0 (Original weights before any training/pruning)
        theta_0_weight = torch.randn(2, 4).detach()
        
        dummy_state_dict = {
            'agent': {
                'network': {
                    # Encoder (unpruned) - note: keys inside 'network' don't have 'network.' prefix in artifact
                    # But wait, in the artifact inspection, keys were: 'encoder.network.0.bias'
                    # My builder logic takes agent['network'] and passes it to _load_module_with_masks.
                    # _load_module_with_masks(prefix='encoder.')
                    # It looks for keys starting with 'encoder.' in source.
                    # So source keys MUST be 'encoder.xxx'.
                    
                    'encoder.network.0.weight': torch.randn(32, 4, 8, 8).detach(),
                    'encoder.network.0.bias': torch.randn(32).detach(),
                    
                    # Head (pruned)
                    'head.network.weight_orig': orig,
                    'head.network.weight_mask': mask,
                    'head.network.bias': torch.zeros(2).detach()
                }
            },
            'theta_0': {
                'network': {
                    'head.network.weight': theta_0_weight
                }
            }
        }
        
        
        # Save dummy state dict to real file to avoid "graph node" deepcopy issues with mocks
        test_ckpt_path = "/tmp/rlp_test_init_ckpt.pt"
        torch.save(dummy_state_dict, test_ckpt_path)
        
        # Mock Checkpointer download to return our real file
        MockCheckpointer.download_artifact_by_path.return_value = (test_ckpt_path, 500, {})
        
        try:
            # We need to mock build_envs to return correct shape info 
            # or just call build_agent directly if we skip training context
            
            # Use DummyEnv
            builder.build_envs = MagicMock(return_value=DummyEnv())
            
            # Build Context (this calls build_agent -> _apply_initial_artifact)
            
            # Use DummyLogger
            ctx = builder.build_training_context(DummyLogger())
            agent = ctx.agent
            pruner = ctx.agent.pruner 
            
            # --- VERIFICATIONS ---
            
            # 1. Spasity Check
            # Head should be pruned
            head_linear = agent.network.head.network
            assert prune.is_pruned(head_linear), "Head linear layer should be pruned"
            
            # Check mask values (Mask 0/1 usually exact, but safer)
            assert torch.allclose(head_linear.weight_mask, mask), "Mask should match artifact"
            assert torch.allclose(head_linear.weight_orig, orig), "Weight_orig should match artifact"
            
            # FORCE SYNC: The 'weight' attribute is stale after load_state_dict because it's not a parameter.
            # We must run a forward pass (or call the hook) to update it.
            dummy_obs = torch.zeros(1, 4, 84, 84)
            agent.network(dummy_obs)
            
            # Check actual weight (should be masked)
            assert torch.allclose(head_linear.weight, weight_pruned), "Weight should be masked"


            # 2. Theta_0 Check
            assert isinstance(pruner, LotteryPruner)
            assert pruner.theta_0 is not None, "LotteryPruner should have theta_0 loaded"
            
            # Check values in theta_0
            # Note: theta_0 keys in artifact were relative to network, 
            # LotteryPruner checks keys relative to agent or network?
            # LotteryPruner._save_theta_0 saves `agent.state_dict()`.
            # So `theta_0` keys should start with `network.`
            
            loaded_theta_0_weight = pruner.theta_0['network']['head.network.weight']
            assert torch.allclose(loaded_theta_0_weight, theta_0_weight), "Theta_0 weights should match"

            
        finally:
            import os
            if os.path.exists(test_ckpt_path):
                os.remove(test_ckpt_path)

