
import pytest
import os
import torch
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
from rlp.core.builder import Builder
from rlp.core.checkpointer import Checkpointer

class TestArtifactInitialization:
    
    @patch("rlp.core.checkpointer.wandb")
    def test_download_artifact_by_path(self, mock_wandb):
        # Setup mock
        mock_api = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.id = "test_artifact_id"
        mock_artifact.download.return_value = "/tmp/downloaded/artifact"
        # MagicMock has all attributes by default. 
        # To test precedence, let's explicitly set history_step to an integer
        mock_artifact.history_step = 12345
        mock_artifact.metadata = {} 
        
        mock_api.artifact.return_value = mock_artifact
        mock_wandb.Api.return_value = mock_api
        
        # Mock finding file
        with patch("rlp.core.checkpointer._latest_in_dir") as mock_latest:
            mock_latest.return_value = "/tmp/downloaded/artifact/model.pt"
            
            # Test
            result = Checkpointer.download_artifact_by_path("entity/project/artifact:v0")
            
            # Verify
            assert result is not None
            filepath, step, metadata = result
            assert filepath == "/tmp/downloaded/artifact/model.pt"
            assert step == 12345
            
            mock_api.artifact.assert_called_with("entity/project/artifact:v0")
            mock_artifact.download.assert_called()

    @patch("rlp.core.builder.Checkpointer")
    def test_apply_initial_artifact(self, MockCheckpointer):
        # 1. Setup Config
        config = OmegaConf.create({
            'initial_artifact': 'entity/project/artifact:v0',
            'algorithm': {
                'name': 'dqn',
                'network': {'encoder': 'nature_cnn', 'head': 'linear', 'hidden_dim': 512},
                'optimizer': {'lr': 1e-4, 'eps': 1e-8},
                'gamma': 0.99, 'tau': 1.0, 'target_network_frequency': 1000,
                'learning_starts': 100, 'train_frequency': 4, 'batch_size': 32,
                'epsilon': {'start': 1.0, 'end': 0.1, 'decay_fraction': 0.1}
            },
            'train': {'total_timesteps': 1000, 'buffer_size': 10000, 'checkpoint_interval': 1000, 'log_interval': 100},
            'pruning': {'method': 'none'},
            'env': {'id': 'PongNoFrameskip-v4', 'num_envs': 1, 'capture_video': False, 'frame_stack': 4},
            'seed': 42,
            'wandb': {'enabled': False}, # Mocked anyway
            'output_dir': '/tmp/test_output'
        })
        
        builder = Builder(config)
        
        # 2. Mock Network and Checkpointer
        mock_network = MagicMock()
        mock_network.encoder = MagicMock()
        mock_network.head = MagicMock()
        
        # Configure load_state_dict return values to behave like real method
        mock_network.encoder.load_state_dict.return_value = ([], [])
        mock_network.head.load_state_dict.return_value = ([], [])
        
        # Checkpointer mock return
        MockCheckpointer.download_artifact_by_path.return_value = ("/tmp/mock.pt", 100, {})
        
        # 3. Mock Torch Load
        # We need to construct a state dict that has masks
        # Key format: "encoder.0.weight_orig", "encoder.0.weight_mask"
        
        dummy_state_dict = {
            "network": {
                "encoder.cnn.0.weight": torch.randn(32, 4, 8, 8), # Unpruned
                "head.fc.weight_orig": torch.randn(512, 10),    # Pruned
                "head.fc.weight_mask": torch.ones(512, 10),
            }
        }
        
        with patch("torch.load", return_value=dummy_state_dict):
             # Mock prune.identity to verify it is called
             with patch("torch.nn.utils.prune.identity") as mock_prune:
                 
                 # Set up named_modules for mock head to simulate finding a pruned module
                 # We need the head to have a submodule named 'fc'
                 mock_fc = MagicMock()
                 mock_network.head.named_modules.return_value = [("fc", mock_fc)]
                 mock_network.encoder.named_modules.return_value = [] # No pruned modules in encoder for this test
                 
                 # Run
                 builder._apply_initial_artifact(mock_network, config.initial_artifact)
                 
                 # Verify Checkpointer called
                 MockCheckpointer.download_artifact_by_path.assert_called_with('entity/project/artifact:v0')
                 
                 # Verify Pruning mask restoration
                 # Should fail if logic didn't see weight_mask
                 # In our dummy dict, "head.fc.weight_mask" exists.
                 # Prefix passed to head load is "head."
                 # Full name = "head.fc"
                 # Key = "head.fc.weight_mask" -> Exists!
                 
                 mock_prune.assert_called_with(mock_fc, 'weight')
                 
                 # Verify load_state_dict called logic
                 # logic calls module.load_state_dict
                 assert mock_network.encoder.load_state_dict.called
                 assert mock_network.head.load_state_dict.called

