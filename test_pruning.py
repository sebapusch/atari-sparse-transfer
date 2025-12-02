import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from rlp.pruning.pruner import GMPPruner, LTHPruner
from rlp.pruning.scheduler import LinearScheduler, CubicScheduler

def test_gmp():
    print("Testing GMP Pruner...")
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    
    # Initialize weights to known values
    nn.init.constant_(model[0].weight, 1.0)
    nn.init.constant_(model[2].weight, 2.0)
    
    # 50% sparsity at step 100
    scheduler = CubicScheduler(0.0, 0.5, 0, 100)
    pruner = GMPPruner(scheduler, update_frequency=10)
    
    # Step 0: 0% sparsity
    pruner.update(model, 0)
    print(f"Step 0 Sparsity: {calculate_sparsity(model):.2f}")
    
    # Step 50: ~6% sparsity (cubic)
    pruner.update(model, 50)
    print(f"Step 50 Sparsity: {calculate_sparsity(model):.2f}")
    
    # Step 100: 50% sparsity
    pruner.update(model, 100)
    print(f"Step 100 Sparsity: {calculate_sparsity(model):.2f}")
    
    # Check if masks exist
    if hasattr(model[0], "weight_mask"):
        print("GMP: weight_mask found (Success)")
    else:
        print("GMP: weight_mask NOT found (Failure)")

def test_lth():
    print("\nTesting LTH Pruner...")
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    
    # 20% pruning per round, every 10 steps
    scheduler = LinearScheduler(0.0, 0.8, 0, 100) # Not used directly for LTH usually, but let's see implementation
    # LTH usually prunes p% of remaining weights.
    # We'll assume LTHPruner takes a rate and frequency.
    
    pruner = LTHPruner(pruning_rate=0.2, pruning_steps=[10, 20, 30])
    
    # Step 0
    pruner.update(model, 0)
    print(f"Step 0 Sparsity: {calculate_sparsity(model):.2f}")
    
    # Step 10: 20%
    pruner.update(model, 10)
    print(f"Step 10 Sparsity: {calculate_sparsity(model):.2f}")
    
    # Step 20: 20% of remaining 80% = 16% + 20% = 36% total? Or 0.8 * 0.8 = 0.64 remaining -> 36% sparsity
    pruner.update(model, 20)
    print(f"Step 20 Sparsity: {calculate_sparsity(model):.2f}")

def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight_mask"):
                mask = module.weight_mask
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()
            else:
                total_params += module.weight.numel()
                # If no mask, assume dense
                pass
    return zero_params / total_params if total_params > 0 else 0.0

if __name__ == "__main__":
    try:
        test_gmp()
    except Exception as e:
        print(f"GMP Test Failed: {e}")
        
    try:
        test_lth()
    except Exception as e:
        print(f"LTH Test Failed: {e}")
