
import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from dataclasses import dataclass

from rlp.pruning.lottery import Lottery, LotteryConfig
from rlp.pruning.utils import calculate_sparsity

# Mocks
class MockNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10)
        )
        self.head = nn.Linear(10, 2)

class MockAgent:
    def __init__(self):
        self.network = MockNetwork()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01)
        
    def state_dict(self):
        return self.network.state_dict() # Simplified

    def finished_training(self, step):
        pass

class MockCtx:
    def __init__(self):
        self.agent = MockAgent()
        self.device = "cpu"
        self.logger = MagicMock()
        self.envs = MagicMock()
        self.buffer = MagicMock()

class MockTrainer:
    def __init__(self):
        self.ctx = MockCtx()
        self.config = MagicMock()
        self.checkpointer = MagicMock()
        self.checkpointer.checkpoint_dir = "/tmp"
        self.external_state = {}
        self.start_step = 0
        
    def train(self):
        pass

class TestLotterySparsity(unittest.TestCase):
    def test_initial_sparsity_calculation(self):
        trainer = MockTrainer()
        
        # Manually prune to 20%
        # Total params = 10*10 + 10 + 10*10 + 10 = 220 (weights + bias)
        # Actually pruning usually only weights. Weights: 100 + 100 = 200.
        # Prune 40 weights -> 20% sparsity
        
        encoder = trainer.ctx.agent.network.encoder
        prune.l1_unstructured(encoder[0], name="weight", amount=0.2)
        prune.l1_unstructured(encoder[1], name="weight", amount=0.2)
        
        # Check sparsity
        current = calculate_sparsity(encoder)
        self.assertAlmostEqual(current, 0.2, places=4)
        
        # Init Lottery
        config = LotteryConfig(final_sparsity=0.5, num_rounds=2)
        lottery = Lottery(trainer, config)
        
        self.assertAlmostEqual(lottery.initial_sparsity, 0.2, places=4)
        
        # Check retention calculation
        # Initial = 0.2, Final = 0.5. Remaining: 0.8 -> 0.5. (Target Ratio: 0.625)
        # 2 rounds. Retention per round = sqrt(0.625) ≈ 0.79057
        
        expected_retention = (0.5 / 0.8) ** 0.5
        self.assertAlmostEqual(lottery.retention_per_round, expected_retention, places=4)
        
        # Check prune_amount calculation
        expected_prune_amount = 1.0 - expected_retention
        self.assertAlmostEqual(lottery.prune_amount, expected_prune_amount, places=4)
        
        # Run Round 1
        # We prune 'prune_amount' of remaining weights.
        # Current Sparsity: 0.2
        # Target Sparsity = 1 - (0.8 * retention) = 1 - 0.6324 = 0.3675
        
        target_sparsity_round_1 = 1.0 - (0.8 * expected_retention)
        
        # Execute Pruning manually with calculated amount
        lottery._prune_network(amount=lottery.prune_amount)
        
        new_sparsity = calculate_sparsity(encoder)
        # We expect ~0.3675
        self.assertAlmostEqual(new_sparsity, target_sparsity_round_1, delta=0.01)
        
        # Run Round 2
        # Start Sparsity: ~0.3675
        # Remaining: ~0.6324
        # Prune 'prune_amount' of remaining -> Target 0.5
        
        lottery._prune_network(amount=lottery.prune_amount)
        new_sparsity = calculate_sparsity(encoder)
        self.assertAlmostEqual(new_sparsity, 0.5, delta=0.01)

if __name__ == '__main__':
    unittest.main()
