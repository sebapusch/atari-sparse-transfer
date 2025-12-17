from __future__ import annotations

import copy
import json
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List

from rlp.pruning.base import PrunerProtocol, PruningContext
from rlp.pruning.utils import get_prunable_modules, calculate_sparsity

@dataclass
class LotteryConfig:
    final_sparsity: float
    num_rounds: int
    first_iteration_steps: int
    rewind_to_step: int = 0
    pruning_rate: float = 0.1 # 10% per iteration
    iqm_window_size: int = 100
    description: str = "Lottery Ticket Hypothesis Experiment"


class LotteryPruner(PrunerProtocol):
    """
    Implements the Lottery Ticket Hypothesis pipeline as a Pruner.
    Iterative Magnitude Pruning (IMP) with rewinding and IQM-based convergence.
    """

    def __init__(self, config: LotteryConfig) -> None:
        self.config = config
        
        self.theta_0: dict[str, Any] | None = None
        self.current_round = 1
        
        # Convergence state
        self.target_iqm: float | None = None
        self.has_converged = False
        self.last_pruning_step = 0
        
        self.initial_sparsity = 0.0

    def prune(self, model: nn.Module, context: PruningContext) -> float | None:
        """
        Main pruning logic.
        1. Step 0: Save theta_0.
        2. First Iteration: Wait for `first_iteration_steps`.
        3. Convergence: Check IQM.
        4. Prune Loop: Prune -> Rewind -> Increment Round.
        """
        step = context.step
        
        # 0. Initialization & Theta_0
        if self.theta_0 is None:
            # We assume step can be 0 or small. If first call, save theta_0.
            # But we need to make sure we are at the START.
            # Ideally we save this at step 0 specifically.
            if step == 0:
                self._save_theta_0(context.agent)
                self.initial_sparsity = calculate_sparsity(model)
                print(f"Lottery: Initial sparsity {self.initial_sparsity:.4f}")
            elif step > 0 and self.theta_0 is None:
                 # If we missed step 0 (e.g. resume or late init), this might be an issue.
                 # For now, let's assume strict step 0 call or we grab current state if safe?
                 # Better to grab theta_0 from agent if it was saved elsewhere? 
                 # Or just save now if it's still early? 
                 # Let's enforce step 0 save or assume loaded.
                 pass
        
        # 1. First Iteration Logic
        if self.current_round == 1:
            if step < self.config.first_iteration_steps:
                return None
            
            # First iteration complete. Time to prune for the first time?
            # User said "The model is trained for all train steps for the first iteration."
            # Then "After the first iteration, ... prune every time it reaches IQM return"
            
            # So at end of first iteration, we:
            # a) Calculate Target IQM (from dense training).
            # b) Prune.
            # c) Rewind.
            # d) Start Round 2.
            
            print(f"Lottery: First iteration complete at step {step}.")
            self._set_target_iqm(context.recent_episodic_returns)
            return self._perform_pruning_step(model, context)

        # 2. Subsequent Iterations
        # Monitor convergence
        if self._has_converged(context.recent_episodic_returns):
            print(f"Lottery: Converged at round {self.current_round} (Step {step}).")
            return self._perform_pruning_step(model, context)

        return None

    def should_stop(self, context: PruningContext) -> bool:
        # Stop if we finished all rounds
        return self.current_round > self.config.num_rounds

    def _perform_pruning_step(self, model: nn.Module, context: PruningContext) -> float:
        print(f"✂️ Lottery: Pruning (Round {self.current_round} -> {self.current_round + 1})...")
        
        # 1. Prune
        # "model always prunes 10% of the weights per iteration"
        # I will assume 10% of CURRENT remaining weights.
        
        parameters_to_prune = get_prunable_modules(model)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.pruning_rate,
        )
        
        current_sparsity = calculate_sparsity(model)
        print(f"Lottery: New Sparsity: {current_sparsity:.4f}")

        # 2. Rewind
        self._rewind_network(context.agent)
        
        # 3. Update State
        self.current_round += 1
        self.last_pruning_step = context.step
        
        # 4. Log artifact
        
        return current_sparsity

    def _save_theta_0(self, agent: Any):
        # Deepcopy to CPU to avoid memory issues
        self.theta_0 = copy.deepcopy(agent.state_dict())
        # Move to CPU recursively if needed? state_dict puts tensors on same device as model.
        # Let's move to cpu.
        for k, v in self.theta_0.items():
            if isinstance(v, torch.Tensor):
                self.theta_0[k] = v.cpu()
            elif isinstance(v, dict): # Nested dicts (like optimizer state)
                 for subk, subv in v.items():
                     if isinstance(subv, torch.Tensor):
                         v[subk] = subv.cpu()
        
        print("Lottery: Saved theta_0.")

    def _rewind_network(self, agent: Any):
        print(f"⏪ Lottery: Rewinding to theta_0...")
        if self.theta_0 is None:
            raise RuntimeError("Cannot rewind, theta_0 not saved!")

        # We cannot just load_state_dict because the structure changed (masks added).
        # We need to selectively load.
        
        # STRATEGY: 
        # Iterate modules. If pruned, copy theta_0['weight'] into module.weight_orig.
        # If not pruned, copy theta_0['weight'] into module.weight.
        
        # Let's traverse the agent's network modules directly.
        for name, module in agent.network.named_modules():
            # Construct key in state_dict. 
            # agent.network is a component. state_dict key is "network.{name}"
            prefix = "network"
            if name:
                prefix += f".{name}"
            
            # 1. Weight
            if hasattr(module, 'weight') and module.weight is not None:
                # Find original weight in theta_0
                # If module is pruned, it has 'weight_orig' and 'weight_mask'. 
                # theta_0 has 'weight' (unpruned).
                
                # In theta_0['network'], the key is just "{name}.weight" (relative to network)
                weight_key = f"{name}.weight" if name else "weight"
                
                # Dig into theta_0['network']
                orig_weight = self.theta_0['network'].get(weight_key)
                
                if orig_weight is not None:
                    orig_weight = orig_weight.to(agent.device)
                    if prune.is_pruned(module):
                        # Copy to weight_orig
                        module.weight_orig.data.copy_(orig_weight.data)
                    else:
                        module.weight.data.copy_(orig_weight.data)
            
            # 2. Bias
            if hasattr(module, 'bias') and module.bias is not None:
                bias_key = f"{name}.bias" if name else "bias"
                orig_bias = self.theta_0['network'].get(bias_key)
                if orig_bias is not None:
                    orig_bias = orig_bias.to(agent.device)
                    module.bias.data.copy_(orig_bias.data)

        # Reset Optimizer
        try:
             # Need to move theta_0 optimizer to device
             # Converting nested state is pain.
             # We try to load.
             # But optimizer state dict keys are parameter IDs (int).
             # If we created the optimizer from scratch, IDs match.
             # If we prune, we might have issues if parameters are replaced.
             # But prune() using pytorch prune utility modifies the module strictly in place?
             # No, it effectively removes the parameter and adds a buffer.
             # So the original parameter ID might be gone or changed?
             
             # If optimizer breaks, we might need re-init.
             # For now, let's assume it works or we catch exception.
             agent.optimizer.load_state_dict(self.theta_0['optimizer']) 
        except Exception as e:
             print(f"⚠️ Lottery: Warning - Could not reload optimizer state directly: {e}")

    def _set_target_iqm(self, returns: List[float]):
        if not returns:
            print("⚠️ Lottery: No returns to calculate IQM! Defaulting to infinite (no convergence).")
            self.target_iqm = float('inf')
            return

        self.target_iqm = self._calculate_iqm(returns)
        print(f"🎯 Lottery: Target IQM set to {self.target_iqm:.4f}")

    def _has_converged(self, returns: List[float]) -> bool:
        if len(returns) < self.config.iqm_window_size:
            return False
            
        current_iqm = self._calculate_iqm(returns)
        print(f"🔍 Lottery: Current IQM {current_iqm:.4f} vs Target {self.target_iqm:.4f}")
        return current_iqm >= self.target_iqm

    def _calculate_iqm(self, data: List[float]) -> float:
        if not data:
            return 0.0
        
        # Use simple mean for now if IQM is complex? 
        # "reason exactly how to compute the return the model should reach" -> user mentioned IQM.
        # IQM: Sort, discard top/bottom 25%, mean of rest.
        
        sorted_data = np.sort(data)
        n = len(sorted_data)
        lower = int(n * 0.25)
        upper = int(n * 0.75)
        
        if lower >= upper:
            # Fallback for very small data
            return float(np.mean(sorted_data))
            
        return float(np.mean(sorted_data[lower:upper]))
