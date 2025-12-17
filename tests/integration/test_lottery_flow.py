import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock
from rlp.pruning.lottery import LotteryPruner, LotteryConfig
from rlp.pruning.base import PruningContext
from rlp.agent.dqn import DQNAgent, DQNConfig
from rlp.components.network import QNetwork
from rlp.components.encoders import NatureCNN
from rlp.components.heads import LinearHead

def test_lottery_pruner_flow():
    # 1. Setup
    config = LotteryConfig(
        final_sparsity=0.5, # Won't be used directly by pruner logic (fixed rate), but good for consistent config
        num_rounds=2,
        first_iteration_steps=10,
        pruning_rate=0.2, # Prune 20%
        iqm_window_size=5
    )
    pruner = LotteryPruner(config)
    
    # Mock Agent & Network
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
    optimizer = torch.optim.Adam(network.parameters())
    
    # We need a fake agent with state_dict
    agent = MagicMock()
    agent.network = network
    agent.optimizer = optimizer
    agent.device = torch.device("cpu")
    agent.state_dict.return_value = {
        "network": {
             "encoder.0.weight": network.encoder[0].weight.detach().clone(),
             "encoder.0.bias": network.encoder[0].bias.detach().clone(),
             "head.weight": network.head.weight.detach().clone(),
             "head.bias": network.head.bias.detach().clone()
        },
        "optimizer": optimizer.state_dict()
    }
    
    # Save theta_0 implicitly called at step 0?
    # Pruner needs step 0 context
    ctx_0 = PruningContext(step=0, agent=agent, recent_episodic_returns=[])
    pruner.prune(network.encoder, ctx_0)
    
    assert pruner.theta_0 is not None
    print("✅ Theta_0 saved.")
    
    # 2. Run First Iteration (Steps 1-9)
    # Should NOT prune
    for step in range(1, 10):
        ctx = PruningContext(step=step, agent=agent, recent_episodic_returns=[1.0, 1.0])
        sparsity = pruner.prune(network.encoder, ctx)
        assert sparsity is None
    
    print("✅ First iteration steps passed without pruning.")
    
    # 3. End of First Iteration (Step 10)
    # Should set target IQM and prune for first time
    # Returns need to be sufficient for IQM (fake it)
    returns = [10.0] * 10
    ctx_10 = PruningContext(step=10, agent=agent, recent_episodic_returns=returns)
    
    # Mock rewind
    agent.state_dict.return_value["network"]["encoder.0.weight"] = network.encoder[0].weight.detach().clone() # Update current state
    
    sparsity = pruner.prune(network.encoder, ctx_10)
    
    assert sparsity is not None
    assert pruner.current_round == 2
    assert pruner.target_iqm == 10.0
    print(f"✅ Round 1 Pruning done. Sparsity: {sparsity}, Target IQM: {pruner.target_iqm}")
    
    # 4. Run Second Iteration
    # Need to reach convergence (IQM >= 10.0)
    # Step 11: returns low
    ctx_11 = PruningContext(step=11, agent=agent, recent_episodic_returns=[5.0]*10)
    sparsity = pruner.prune(network.encoder, ctx_11)
    assert sparsity is None
    
    # Step 12: returns high (converged)
    ctx_12 = PruningContext(step=12, agent=agent, recent_episodic_returns=[10.0]*10)
    sparsity = pruner.prune(network.encoder, ctx_12)
    
    assert sparsity is not None
    assert pruner.current_round == 3
    print(f"✅ Round 2 Pruning done (Convergence). Sparsity: {sparsity}")
    
    # 5. Check Stop Condition
    # config num_rounds = 2. We are at round 3 (finished 2 rounds).
    # so should_stop should be True?
    # Logic: current_round starts at 1.
    # After round 1 prune -> current_round = 2.
    # After round 2 prune -> current_round = 3.
    # If 3 > 2 -> True.
    assert pruner.should_stop(ctx_12)
    print("✅ Stopping condition met.")

if __name__ == "__main__":
    test_lottery_pruner_flow()
