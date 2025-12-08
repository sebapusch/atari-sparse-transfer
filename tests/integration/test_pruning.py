import copy
import torch
import torch.nn as nn
import torch.optim as optim
from rlp.agent.dqn import DQNAgent, DQNConfig
from rlp.components.network import QNetwork
from rlp.pruning.pruner import GMPPruner
from rlp.pruning.scheduler import ConstantScheduler
from rlp.pruning.utils import calculate_sparsity

# Mock Network
class MockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50) # Increased size to reduce rounding errors in sparsity
        self.fc2 = nn.Linear(50, 50)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class MockHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        return self.fc3(x)

def test_target_network_sparsity_sync():
    # Setup
    encoder = MockEncoder()
    head = MockHead()
    network = QNetwork(encoder, head)
    
    # Use tau=1.0 to ensure values are copied exactly if logic works
    config = DQNConfig(num_actions=2, gamma=0.99, tau=1.0) 
    
    optimizer = optim.Adam(network.parameters())
    agent = DQNAgent(network, optimizer, config, device="cpu")
    
    # Determine initial sparsity (should be 0)
    assert calculate_sparsity(agent.network) == 0.0
    
    # Prune network to 0.8
    # We use ConstantScheduler(0.8) which requests 0.8
    scheduler = ConstantScheduler(0.8)
    pruner = GMPPruner(scheduler, update_frequency=1)
    
    # Run prune
    # Step=1 matches frequency=1
    new_sparsity = pruner.prune(agent.network, step=1)
    
    # Check sparsity matches target
    assert new_sparsity >= 0.79 # Approx check
    print(f"Network Sparsity: {new_sparsity}")
    
    # Verify current sparsity
    assert calculate_sparsity(agent.network) == new_sparsity
    
    # Verify target network is dense (0.0) before update (it was deepcopied before pruning)
    assert calculate_sparsity(agent.target_network) == 0.0
    
    # Update target network
    agent._update_target_network()
    
    # Check target network sparsity (BY VALUE)
    # Since target network is NOT pruned (no masks), calculate_sparsity would return 0.
    # We must manually count zeros to verify values were copied.
    
    total_params = 0
    zero_params = 0
    for module in agent.target_network.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                total_params += w.numel()
                zero_params += (w == 0).sum().item()
            # bias often not pruned so we skip check or include it? 
            # GMPPruner usually prunes weights only.
    
    target_value_sparsity = zero_params / total_params if total_params > 0 else 0.0
    print(f"Target Network Value Sparsity: {target_value_sparsity}")
    
    # This assertion ensures that the target network has acquired the sparsity structure (values)
    # even though it is physically dense.
    assert abs(target_value_sparsity - new_sparsity) < 0.01, f"Target sparsity {target_value_sparsity} != {new_sparsity}"

    # Additionally check that the values in target network respect the mask
    # (Since we used tau=1.0, they should be exactly the source masked values)
    # We can perform a deeper check if needed, but sparsity check covers the mask existence.

if __name__ == "__main__":
    try:
        test_target_network_sparsity_sync()
        print("Test PASSED")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"Test CRASHED: {e}")
        import traceback
        traceback.print_exc()
