
import pytest
from unittest.mock import MagicMock, patch
from rlp.pruning.lottery import Lottery, LotteryConfig
from rlp.core.trainer import Trainer
from rlp.agent.base import AgentProtocol
import torch
import os

class TestLotteryIntegration:
    
    @pytest.fixture
    def mock_trainer(self):
        trainer = MagicMock(spec=Trainer)
        trainer.cfg = MagicMock()
        trainer.cfg.seed = 42
        trainer.ctx = MagicMock()
        trainer.ctx.device = torch.device('cpu')
        trainer.ctx.agent = MagicMock(spec=AgentProtocol)
        trainer.ctx.agent.network = torch.nn.Linear(10, 10)
        trainer.ctx.agent.optimizer = MagicMock()
        trainer.ctx.agent.optimizer.state_dict.return_value = {'opt': 'state'}
        trainer.ctx.agent.state_dict.return_value = {'network': {'weight': torch.randn(10, 10)}}
        
        trainer.checkpointer = MagicMock()
        trainer.checkpointer.checkpoint_dir = "/tmp/ckpt"
        trainer.external_state = {}
        return trainer

    def test_lottery_init_fresh(self, mock_trainer):
        config = LotteryConfig(final_sparsity=0.5, num_rounds=2)
        
        # Test fresh init saves theta_0
        with patch("torch.save") as mock_save, \
             patch("os.path.join", return_value="/tmp/ckpt/theta_0.pt"):
            lottery = Lottery(mock_trainer, config)
            
            assert lottery.start_round == 1
            # Should save theta_0
            mock_save.assert_called()
            args, _ = mock_save.call_args
            # Verify structure but don't compare tensors with == as it fails
            assert 'agent' in args[0]
            assert 'optimizer' in args[0]

    def test_lottery_run_logic(self, mock_trainer):
        config = LotteryConfig(final_sparsity=0.5, num_rounds=2)
        
        with patch("torch.save"), patch("os.path.join", return_value="/tmp/ckpt/theta_0.pt"):
             lottery = Lottery(mock_trainer, config)
        
        with patch.object(lottery, "_prune_network") as mock_prune, \
             patch.object(lottery, "_rewind_network") as mock_rewind, \
             patch.object(lottery, "_log_artifacts") as mock_log, \
             patch.object(lottery, "_save_theta_0"), \
             patch("rlp.pruning.lottery.calculate_sparsity", return_value=0.5): 
             
             lottery.run()
             
             assert mock_trainer.train.call_count == 2
             assert mock_prune.call_count == 2
             assert mock_rewind.call_count == 2
             
             # Check external state updates
             assert mock_trainer.external_state['lottery_round'] == 2

    def test_lottery_resume(self, mock_trainer):
        config = LotteryConfig(final_sparsity=0.5, num_rounds=3)
        resume_state = {'lottery_round': 2}
        
        with patch("torch.load") as mock_load, \
             patch("os.path.exists", return_value=True), \
             patch("os.path.join", return_value="/tmp/ckpt/theta_0.pt"):
            
            mock_load.return_value = {
                "agent": {'network': {'weight': torch.randn(10, 10)}}, # rewound state
                "optimizer": {'opt': 'rewound'}
            }
            
            lottery = Lottery(mock_trainer, config, resume_state=resume_state)
            
            assert lottery.start_round == 2
            
            with patch.object(lottery, "_prune_network"), \
                 patch.object(lottery, "_rewind_network"), \
                 patch.object(lottery, "_log_artifacts"), \
                 patch("rlp.pruning.lottery.calculate_sparsity", return_value=0.5):
                
                 lottery.run()
                 
                 # Should run round 2 and 3
                 assert mock_trainer.train.call_count == 2
                 assert mock_trainer.external_state['lottery_round'] == 3
