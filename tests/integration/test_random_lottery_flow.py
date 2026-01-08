import torch
import torch.nn as nn
import numpy as np
import copy
from unittest.mock import MagicMock
from rlp.pruning.lottery import RandomLotteryPruner, LotteryConfig
from rlp.pruning.base import PruningContext

def test_random_lottery_pruner_flow():
    # 1. Setup
    config = LotteryConfig(
        final_sparsity=0.36, 
        first_iteration_steps=5,
        pruning_rate=0.2, 
        iqm_window_size=2
    )
    pruner = RandomLotteryPruner(config)
    
    # Mock Network
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU()
            )
            self.head = nn.Linear(10, 4)
            
        def forward(self, x):
            return self.head(self.encoder(x))
            
    network = SimpleNet()
    
    # Manually capture initial weights (theta_0 equivalent)
    # We clone to ensure we have a static reference
    initial_weights = copy.deepcopy(network.state_dict())
    
    optimizer = torch.optim.Adam(network.parameters())
    # Mock some optimizer state
    optimizer.state['dummy'] = 'data'
    
    agent = MagicMock()
    agent.network = network
    agent.optimizer = optimizer
    agent.device = torch.device("cpu")
    # For state_dict, we return the current network state
    agent.state_dict.side_effect = lambda: {"network": network.state_dict(), "optimizer": optimizer.state_dict()}
    
    # Step 0: Initialize
    ctx_0 = PruningContext(step=0, agent=agent, recent_episodic_returns=[])
    pruner.prune(network.encoder, ctx_0)
    
    # Verify we didn't crash and theta_0 dummy is set
    assert pruner.theta_0 is not None
    assert pruner.theta_0.get("dummy") is True
    print("✅ Initialization passed.")
    
    # Run iterations until pruning
    # returns needed for convergence
    returns = [10.0] * 5
    
    # Simulate step 5 (end of first iteration)
    ctx_5 = PruningContext(step=5, agent=agent, recent_episodic_returns=returns)
    
    # Before pruning, modify weights to simulate learning
    with torch.no_grad():
        network.encoder[0].weight.add_(1.0)
    
    weights_before_prune = copy.deepcopy(network.state_dict())
    
    # Perform pruning
    sparsity = pruner.prune(network.encoder, ctx_5)
    
    assert sparsity is not None
    assert sparsity > 0.0
    print(f"✅ Pruning occurred. Sparsity: {sparsity}")
    
    # Verify Random Reinitialization
    # 1. Weights should NOT match weights_before_prune (obviously)
    # 2. Weights should NOT match initial_weights (which would happen if we rewound to theta_0)
    # 3. Weights should validly be random (hard to test without statistical check, but inequality is good enough for logic flow)
    
    current_weights = network.state_dict()
    
    # Check Encoder Weight 
    # It is pruned, so we check weight_orig ? 
    # PyTorch state_dict contains 'weight_orig' and 'weight_mask' for pruned params.
    # 'weight' is not in state_dict usually if pruned (it is an attribute).
    # Wait, simple network state_dict with pruning usually has "encoder.0.weight_orig"
    
    enc_weight_orig = current_weights['encoder.0.weight_orig']
    initial_enc_weight = initial_weights['encoder.0.weight'] # Was unpruned then
    
    if torch.allclose(enc_weight_orig, initial_enc_weight):
        print("❌ Error: Weights matching initial weights! It seems we simply rewound to theta_0 or didn't reinit.")
        # Note: chance of random matching initial is zero.
        assert False, "Weights match initial theta_0 - Reinitialization failed."
    else:
        print("✅ Weights differ from initial weights (theta_0).")
        
    # Verify Optimizer Reset
    assert len(optimizer.state) == 0
    print("✅ Optimizer state cleared.")
    
    # Verify Mask Structure (Random Pruning)
    mask = current_weights['encoder.0.weight_mask']
    assert torch.sum(mask) < mask.numel()
    print("✅ Mask present.")

if __name__ == "__main__":
    test_random_lottery_pruner_flow()
