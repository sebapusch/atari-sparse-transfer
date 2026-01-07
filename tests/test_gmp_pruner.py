
import pytest
import torch
import torch.nn as nn
from rlp.pruning.pruner import GMPPruner
from rlp.pruning.scheduler import LinearScheduler
from rlp.pruning.base import PruningContext
from rlp.pruning.utils import calculate_sparsity

def test_gmp_pruning_behavior():
    # 1. Setup Model
    model = nn.Sequential(
        nn.Linear(10, 10, bias=False),  # 100 weights
        nn.Linear(10, 10, bias=False)   # 100 weights
    )
    # Initialize with known weights to predict pruning
    # All weights = 1.0 initially
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)
    
    # Set some weights to smaller values to ensure they are pruned first
    # Set first 10 weights of layer 0 to 0.1
    # 10 weights are 0.1, rest are 1.0. Total 200 weights.
    with torch.no_grad():
        list(model.parameters())[0].data[0, :] = 0.1 

    # 2. Setup Scheduler and Pruner
    # Linearly increase sparsity from 0.0 to 0.5 over 100 steps
    scheduler = LinearScheduler(initial_sparsity=0.0, final_sparsity=0.5, start_step=0, end_step=100)
    update_frequency = 10
    pruner = GMPPruner(scheduler, update_frequency=update_frequency)

    # 3. Simulation Loop
    for step in range(0, 101):
        context = PruningContext(step=step, agent=None)
        
        result_sparsity = pruner.prune(model, context)
        
        current_sparsity = calculate_sparsity(model)
        expected_target_sparsity = scheduler.get_sparsity(step, 0)

        if step % update_frequency == 0:
            # Check if actual sparsity is close to target
            assert abs(current_sparsity - expected_target_sparsity) < 1e-4, \
                f"Sparsity mismatch at step {step}: {current_sparsity} != {expected_target_sparsity}"
            
            # Check if correct weights were pruned (the 0.1 ones)
            # At step 10, target is 0.05 (5% of 200 = 10 params).
            if step == 10:
                # Check if the first row of first layer is zeroed
                is_pruned = (list(model.children())[0].weight[0, :] == 0).all()
                assert is_pruned, "Smallest weights were NOT pruned correctly at step 10"

        else:
            # Ensure no pruning happened (result should be None)
            assert result_sparsity is None, f"Pruning happened at step {step} but shouldn't have"
