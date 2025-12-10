import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from rlp.pruning.lottery import Lottery, LotteryConfig
from rlp.core.trainer import Trainer, TrainingContext, TrainingConfig
import torch.nn.utils.prune as prune

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 10)
        self.head = nn.Linear(10, 2)
        
class MockAgent:
    def __init__(self):
        self.network = SimpleNet()
        self.target_network = SimpleNet()
        self.optimizer = MagicMock()
        self.optimizer.state_dict.return_value = {"opt": "state"}
        
    def state_dict(self):
        return {
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

class TestLottery(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()
        self.ctx = MagicMock(spec=TrainingContext)
        self.ctx.agent = self.agent
        self.ctx.buffer = MagicMock()
        self.ctx.envs = MagicMock()
        self.ctx.logger = MagicMock()
        self.ctx.logger.run = None # Default to None to skip wandb logging in tests
        
        self.cfg = MagicMock(spec=TrainingConfig)
        self.cfg.seed = 42
        
        self.trainer = MagicMock(spec=Trainer)
        self.trainer.ctx = self.ctx
        self.trainer.cfg = self.cfg
        self.trainer.train = MagicMock()
        
        # Lottery Config
        self.lottery_config = LotteryConfig(
            final_sparsity=0.5,
            num_rounds=2
        )
        
        # Init weights to known values
        with torch.no_grad():
            self.agent.network.encoder.weight.fill_(1.0)
            self.agent.network.head.weight.fill_(1.0)
            self.agent.target_network.encoder.weight.fill_(1.0)

    def test_initialization(self):
        lottery = Lottery(self.trainer, self.lottery_config)
        self.assertIsNotNone(lottery.rewind_state)
        # Check pruning rate calculation
        # S_final = 0.5, N=2. 
        # r = (1 - 0.5)^(1/2) = 0.7071
        # p = 1 - 0.7071 = 0.2929
        expected_p = 1.0 - (0.5 ** 0.5)
        self.assertAlmostEqual(lottery.prune_amount, expected_p, places=4)

    def test_pruning_and_rewinding(self):
        lottery = Lottery(self.trainer, self.lottery_config)
        
        # Simulate training changing weights
        with torch.no_grad():
            self.agent.network.encoder.weight.add_(1.0) # become 2.0
            
        # Verify weights changed
        self.assertTrue((self.agent.network.encoder.weight == 2.0).all())
        
        # Run 1 prune step manually
        lottery._prune_network()
        
        # Check masking applied (buffers exist)
        self.assertTrue(hasattr(self.agent.network.encoder, "weight_mask"))
        self.assertTrue(prune.is_pruned(self.agent.network.encoder))
        
        # Run rewind manually
        lottery._rewind_network()
        
        # Check restoration
        # Used-to-be-pruned weights should be 0 (due to mask).
        # Surviving weights should be 1.0 (original value), not 2.0 (trained value).
        
        mask = self.agent.network.encoder.weight_mask
        weight = self.agent.network.encoder.weight
        
        # Check masked values are 0
        self.assertTrue((weight[mask == 0] == 0).all())
        
        # Check surviving values match initialization (1.0)
        # Note: 'weight' attribute is 'weight_orig' * 'mask'.
        # 'weight_orig' should be restored to 1.0.
        
        weight_orig = self.agent.network.encoder.weight_orig
        self.assertTrue((weight_orig == 1.0).all())
        
        # Check optimizer reset
        self.agent.optimizer.load_state_dict.assert_called()

    def test_run_loop(self):
        lottery = Lottery(self.trainer, self.lottery_config)
        
        # Set prune_amount to something simple slightly different to ensure we prune something
        # 0.29
        
        lottery.run()
        
        # Check trainer called twice
        self.assertEqual(self.trainer.train.call_count, 2)
        
        # Check reset called twice
        self.assertEqual(self.trainer.ctx.buffer.reset.call_count, 2)

if __name__ == '__main__':
    unittest.main()
