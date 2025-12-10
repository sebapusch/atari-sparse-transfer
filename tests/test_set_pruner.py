import pytest
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from rlp.pruning.set_pruner import SETPruner
from rlp.pruning.utils import calculate_sparsity

def test_set_pruner_initialization():
    """Test that SET initialization creates sparse masks."""
    model = nn.Sequential(
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    
    # Large epsilon means low sparsity (dense). Small epsilon means high sparsity.
    # p = epsilon * (n_in + n_out) / (n_in * n_out)
    # For 100x100: p = 20 * 200 / 10000 = 0.4 -> Sparsity 0.6
    pruner = SETPruner(epsilon=20, zeta=0.3, update_frequency=10)
    
    # Step 0 should initialize
    model_sparsity = pruner.prune(model, 0)
    
    assert model_sparsity is not None
    assert 0.0 < model_sparsity < 1.0
    
    # Check if linear layers are pruned
    assert prune.is_pruned(model[0])
    assert prune.is_pruned(model[2])
    
    # Check approximate sparsity of layer 0
    # Expected density ~0.4, so sparsity ~0.6
    mask = model[0].weight_mask
    sparsity = (mask == 0).float().mean().item()
    assert 0.5 < sparsity < 0.7  # Allow for some variance due to randomness

def test_constant_sparsity_evolution():
    """Verify sparsity remains constant after evolution."""
    model = nn.Sequential(nn.Linear(100, 100, bias=False))
    pruner = SETPruner(epsilon=10, zeta=0.2, update_frequency=1)
    
    # Init
    pruner.prune(model, 0)
    initial_sparsity = calculate_sparsity(model)
    initial_active = int(model[0].weight_mask.sum().item())
    
    # Evolve
    # Step 1 matches update_frequency=1
    pruner.prune(model, 1)
    
    final_sparsity = calculate_sparsity(model)
    final_active = int(model[0].weight_mask.sum().item())
    
    assert initial_active == final_active
    assert abs(initial_sparsity - final_sparsity) < 1e-6

def test_evolution_topology_change():
    """Verify that the mask actually changes (topology evolves)."""
    model = nn.Sequential(nn.Linear(50, 50, bias=False))
    pruner = SETPruner(epsilon=10, zeta=0.5, update_frequency=1)
    
    pruner.prune(model, 0)
    mask1 = model[0].weight_mask.clone()
    
    # Manually set some weights to be small to ensure they get pruned
    # We need to modify weight_orig and ensure weight reflects it
    # But weight is weight_orig * mask. 
    # For test, we can just let it prune random small weights initialized.
    
    pruner.prune(model, 1)
    mask2 = model[0].weight_mask.clone()
    
    # Verify masks are different
    assert not torch.all(mask1 == mask2)
    
    # Verify same number of active connections
    assert mask1.sum().item() == mask2.sum().item()

def test_reinitialization():
    """Verify that new connections are reinitialized."""
    model = nn.Sequential(nn.Linear(50, 50, bias=False)) 
    pruner = SETPruner(epsilon=10, zeta=0.5, update_frequency=1)
    
    pruner.prune(model, 0)
    
    # Force some weights to be zero in weight_orig to verify re-init works
    # (Though SET re-inits whatever was there)
    
    # Let's track indices that were 0 in mask1 and became 1 in mask2
    mask1 = model[0].weight_mask.clone()
    weight_orig1 = model[0].weight_orig.clone()
    
    pruner.prune(model, 1)
    mask2 = model[0].weight_mask.clone()
    weight_orig2 = model[0].weight_orig.clone()
    
    # Find regrown indices: mask1==0 and mask2==1
    regrown = (mask1 == 0) & (mask2 == 1)
    
    assert regrown.sum() > 0
    
    # Check that weight_orig changed at these indices
    # (There is a tiny chance random init equals previous value, but negligible)
    assert not torch.all(weight_orig1[regrown] == weight_orig2[regrown])
    
    # Check that weights are non-zero
    assert torch.all(weight_orig2[regrown] != 0)

def test_no_grad_flow_masked():
    """Verify gradients do not flow through pruned weights."""
    model = nn.Linear(10, 10, bias=False)
    pruner = SETPruner(epsilon=5, zeta=0.1, update_frequency=10)
    
    pruner.prune(model, 0)
    
    x = torch.randn(1, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Gradient should be 0 where mask is 0
    # PyTorch pruning handles this by hook: grad on weight is dense, but we check if result respects it?
    # Actually PyTorch's `weight` parameter is derived. 
    # The gradient we care about is on `weight_orig`?
    # Pruning hook: weight = weight_orig * mask.
    # dL/d(weight_orig) = dL/d(weight) * mask.
    # So yes, gradient on weight_orig should be 0 where mask is 0.
    
    mask = model.weight_mask
    grad = model.weight_orig.grad
    
    masked_grad = grad * (1 - mask)
    assert torch.all(masked_grad == 0)
