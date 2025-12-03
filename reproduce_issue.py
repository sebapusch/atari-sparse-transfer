import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

def reproduce():
    # Create a simple model with Conv2d
    model = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    
    # Create target network (unpruned)
    target_model = copy.deepcopy(model)
    
    print("Original parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
        
    # Apply pruning to model (only Conv2d)
    print("\nApplying pruning to model[0] (Conv2d)...")
    prune.global_unstructured(
        [(model[0], 'weight')],
        pruning_method=prune.L1Unstructured,
        amount=0.5,
    )
    
    print("\nPruned parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
        
    print("\nTarget parameters:")
    for name, param in target_model.named_parameters():
        print(f"{name}: {param.shape}")

    # Fix attempt: Apply identity pruning to target model to match structure
    print("\nApplying identity pruning to target model...")
    # We need to apply the same pruning method structure
    # prune.global_unstructured works by applying pruning to specified parameters.
    # If we use amount=0, it shouldn't prune anything but should restructure.
    prune.global_unstructured(
        [(target_model[0], 'weight')],
        pruning_method=prune.L1Unstructured,
        amount=0.0,
    )
    
    print("\nTarget parameters after identity pruning:")
    for name, param in target_model.named_parameters():
        print(f"{name}: {param.shape}")
        
    # Simulate soft update
    print("\nSimulating soft update...")
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), target_model.named_parameters()):
        print(f"Updating {name1} ({param1.shape}) with {name2} ({param2.shape})")
        try:
            # Soft update logic: param1 is source, param2 is target
            # In code: target.data = tau * source.data + (1-tau) * target.data
            # But wait, the code says:
            # self.tau * param.data + (1.0 - self.tau) * target_param.data
            # param is source (network), target_param is target (target_network)
            # So we are updating target_param with param.
            
            # If param1 is source (pruned) and param2 is target (unpruned)
            res = 0.5 * param1.data + 0.5 * param2.data
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return

if __name__ == "__main__":
    reproduce()
