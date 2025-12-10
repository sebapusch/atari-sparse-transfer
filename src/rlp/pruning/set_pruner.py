from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, Optional, Tuple, List

from rlp.pruning.base import PrunerProtocol
from rlp.pruning.utils import get_prunable_modules, calculate_sparsity


class SETPruner(PrunerProtocol):
    """
    Sparse Evolutionary Training (SET) Pruner.
    
    Implements the SET algorithm from Mocanu et al. (2018).
    - Initializes with Erdős–Rényi topology.
    - Maintains constant sparsity per layer.
    - Periodically removes smallest weights and regrows random new connections.
    """
    
    def __init__(self, epsilon: float = 20, zeta: float = 0.3, update_frequency: int = 100) -> None:
        """
        Args:
            epsilon: Parameter controling sparsity density (larger = denser).
                     Density p = epsilon * (n_in + n_out) / (n_in * n_out)
            zeta: Fraction of weights to prune and regrow at each evolution step.
            update_frequency: Number of steps between evolution cycles.
        """
        self.epsilon = epsilon
        self.zeta = zeta
        self.update_frequency = update_frequency
        self.initialized = False

    def prune(self, model: nn.Module, step: int) -> float | None:
        """
        Apply SET pruning logic.
        
        Args:
            model: The model to prune.
            step: Current training step.
            
        Returns:
            Current global sparsity after operation, or None if no op performed.
        """
        # 1. Initialization Step
        if step == 0 and not self.initialized:
            self._initialize_topology(model)
            self.initialized = True
            return calculate_sparsity(model)

        # 2. Evolution Step
        if step > 0 and step % self.update_frequency == 0:
            self._evolve_topology(model)
            return calculate_sparsity(model)
            
        return None

    def _initialize_topology(self, model: nn.Module) -> None:
        """Perform Erdős-Rényi initialization."""
        prunable_modules = get_prunable_modules(model)
        
        for module, param_name in prunable_modules:
            weight = getattr(module, param_name)
            n_out, n_in = weight.shape[:2]
            
            # Conv2d weights are (out, in, k, k), adjust n_in to be total fan-in per filter
            if isinstance(module, nn.Conv2d):
                k_h, k_w = weight.shape[2], weight.shape[3]
                n_in = n_in * k_h * k_w
                # Total params = n_out * n_in_real
            
            total_params = weight.numel()
            
            # Probability of connection
            # p = epsilon * (n_in + n_out) / (n_in * n_out) is for dense layers approx.
            # For general tensor: epsilon * (dim0 + dim1) / (dim0 * dim1)?
            # The paper defines it for bipartite graphs (Linear layers).
            # For Conv layers, most implementations treat them similarly or use standard sparsity.
            # We will use the formula: p = epsilon * (n_out + n_in) / (n_out * n_in)
            # Note: If p > 1, we cap at 1.
            
            density = self.epsilon * (n_out + n_in) / (n_out * n_in)
            density = min(1.0, max(0.0, density))
            sparsity = 1.0 - density
            
            # Remove existing pruning if any (to start fresh or check if already pruned)
            # But usually we assume fresh model at step 0.
            
            # Using RandomUnstructured to create the initial mask
            # We want to keep `density` fraction, so we prune `sparsity` fraction.
            prune.random_unstructured(module, name=param_name, amount=sparsity)


    def _evolve_topology(self, model: nn.Module) -> None:
        """Perform Prune-and-Regrow cycle."""
        prunable_modules = get_prunable_modules(model)
        
        for module, param_name in prunable_modules:
            if not prune.is_pruned(module):
                continue
                
            # 1. Prune: Remove smallest magnitude weights
            # We need to operate on the 'weight' attribute which is masked.
            # But the mask is stored in 'weight_mask'.
            # We want to set some 1s to 0s in the mask.
            
            mask = getattr(module, param_name + "_mask")
            weight_data = getattr(module, param_name).data # masked weights
            
            # Get Number of active connections
            active_count = int(mask.sum().item())
            n_prune = int(active_count * self.zeta)
            
            if n_prune == 0:
                continue
            
            # Find the indices of the n_prune smallest non-zero weights
            # Flatten weights
            flat_weights = weight_data.view(-1)
            flat_mask = mask.view(-1)
            
            # Get indices where mask is 1
            active_indices = torch.nonzero(flat_mask).view(-1)
            
            # Get weights at active indices
            active_weights = flat_weights[active_indices]
            
            # Get indices of smallest absolute values
            _, sorted_idx = torch.topk(active_weights.abs(), k=n_prune, largest=False)
            prune_indices = active_indices[sorted_idx]
            
            # Update mask: set these to 0
            flat_mask[prune_indices] = 0
            
            # 2. Regrow: Add random connections
            # Find indices where mask is 0
            zero_indices = torch.nonzero(flat_mask == 0).view(-1)
            
            # We need to pick n_prune random indices from zero_indices
            if zero_indices.numel() < n_prune:
                # Should not happen unless simple fully connected
                n_regrow = zero_indices.numel()
            else:
                n_regrow = n_prune
            
            perm = torch.randperm(zero_indices.numel(), device=weight_data.device)
            regrow_indices = zero_indices[perm[:n_regrow]]
            
            # Update mask: set these to 1
            flat_mask[regrow_indices] = 1
            
            # 3. Reinitialize Regrown Weights
            # We need to update `weight_orig` at regrow_indices.
            # If we don't, they will have their old values (which might be 0 or old weights).
            # Paper says: "initialized to random values"
            
            # Sample new values using Kaiming Uniform or Xavier, or standard normal.
            # Let's generate a temporary tensor of same shape initialized with kaiming
            # and copy values.
            
            weight_orig = getattr(module, param_name + "_orig")
            dummy_weight = torch.zeros_like(weight_orig)
            nn.init.kaiming_uniform_(dummy_weight, a=math.sqrt(5)) 
            # Note: Linear default init is kaiming_uniform with a=sqrt(5)
            
            flat_dummy = dummy_weight.view(-1)
            
            # Update weight_orig
            # Note: module.weight_orig is the parameter tensor
            with torch.no_grad():
                weight_orig.view(-1)[regrow_indices] = flat_dummy[regrow_indices]
                
            # No need to explicitly update module.weight logic, 
            # PyTorch's forward hook will apply (weight_orig * mask) -> weight.
            # But we are modifying the mask tensor in-place? 
            # warning: mask is a buffer, we can modify it.
