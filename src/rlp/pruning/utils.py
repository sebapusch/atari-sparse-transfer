from torch import nn
from torch.nn.utils import prune


def get_prunable_modules(module: nn.Module) -> list[tuple[nn.Module, str]]:
    to_prune = []
    for name, module in module.named_modules():
        if hasattr(module, 'weight'):
            to_prune.append((module, 'weight'))
        # if hasattr(module, 'bias') and module.bias is not None:
        #     # Optional: usually we don't prune bias, but if you want to:
        #     to_prune.append((module, 'bias'))
    return to_prune

def calculate_sparsity(module: nn.Module) -> float:
    total_params = 0
    zero_params = 0
    for module, name in get_prunable_modules(module):
        if prune.is_pruned(module):
            mask = getattr(module, name + "_mask")
            total_params += mask.numel()
            zero_params += (mask == 0).sum().item()
        else:
            total_params += getattr(module, name).numel()

    return zero_params / total_params if total_params > 0 else 0.0
