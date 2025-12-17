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
        # Will be calculated once we know initial sparsity (at step 0 or first prune)
        self.total_rounds: int | None = None 

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
            # If theta_0 is not set, we save it now.
            # This handles cases where training starts later (e.g. learning_starts > 0)
            # or if we are just starting the LTH process.
            # We assume the weights at this point are the "initial" weights (theta_0).
            # If using learning_starts, no updates have happened yet, so this is correct.
            self._save_theta_0(context.agent)
            self.initial_sparsity = calculate_sparsity(model)
            self._calculate_total_rounds()
            print(f"Lottery: Initial sparsity {self.initial_sparsity:.4f}")
            print(f"Lottery: Calculated total rounds needed: {self.total_rounds}")
            print(f"Lottery: Theta_0 saved at step {step}.")
        
        # 1. First Iteration Logic
        if self.current_round == 1:
            if step < self.config.first_iteration_steps:
                return None
            
            # Ensure total_rounds is calculated if we missed step 0 
            # (though we should have caught it, but safely recalculate)
            if self.total_rounds is None:
                 self.initial_sparsity = calculate_sparsity(model)
                 self._calculate_total_rounds()
            
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
        if self.total_rounds is None:
            return False
            
        # If we just finished round X, and X == total_rounds
        # current_round is incremented AFTER pruning.
        # So if we want to do N rounds, we are done when current_round > N.
        # HOWEVER, the last round is a training round too?
        # Standard LTH: Train -> Prune -> Rewind -> Train ... until target sparsity.
        # Usually we stop AFTER the training of the final ticket? Or just after finding it?
        # User said: "calculates the number of iterations necessary to reach final_sparsity".
        # Assuming we stop once we have the final ticket (and maybe trained it? or just stop?)
        # "delegate stopping" implies we stop the whole process.
        # If we just pruned to final sparsity, we are at start of training that final ticket.
        # Should we train it? Probably yes.
        # So stop when current_round > total_rounds + 1? Or just explicit check?
        # Let's assume we stop when we exceed rounds.
        return self.current_round > self.total_rounds

    def _calculate_total_rounds(self):
        # Calculate rounds needed.
        # (1 - p)^n = (1 - S_final) / (1 - S_initial)
        # n = log( (1-Sf)/(1-Si) ) / log(1-p)
        
        if self.initial_sparsity >= self.config.final_sparsity:
            self.total_rounds = 0
            return

        numerator = np.log((1.0 - self.config.final_sparsity) / (1.0 - self.initial_sparsity))
        denominator = np.log(1.0 - self.config.pruning_rate)
        
        if denominator == 0:
            self.total_rounds = 0
        else:
            # Add small epsilon to handle floating point exact matches (e.g. 2.0 -> 2)
            # If numerator/denominator is 2.00000000004, ceil makes it 3.
            # We want to be tolerant.
            ratio = numerator / denominator
            self.total_rounds = int(np.ceil(ratio - 1e-9))

    def _perform_pruning_step(self, model: nn.Module, context: PruningContext) -> float:
        print(f"✂️ Lottery: Pruning (Round {self.current_round} -> {self.current_round + 1})...")
        
        # Calculate amount to prune
        current_sparsity = calculate_sparsity(model)
        
        amount = self.config.pruning_rate
        
        # Check if this is the final round
        if self.current_round == self.total_rounds:
            # We want to hit final_sparsity EXACTLY.
            # amount = (target - current) / (1 - current)
            if current_sparsity < self.config.final_sparsity:
                numerator = self.config.final_sparsity - current_sparsity
                denominator = 1.0 - current_sparsity
                if denominator > 0:
                    amount = numerator / denominator
                else:
                    amount = 0.0
            else:
                amount = 0.0
                
            print(f"Lottery: Final Round Correction. Calculated amount: {amount:.4f} to reach {self.config.final_sparsity}")

        # Safety clamp
        amount = max(0.0, min(1.0, amount))

        parameters_to_prune = get_prunable_modules(model)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        
        new_sparsity = calculate_sparsity(model)
        print(f"Lottery: New Sparsity: {new_sparsity:.4f}")

        # 2. Rewind
        self._rewind_network(context.agent)

        context.agent.update_target_network()
        
        # 3. Update State
        self.current_round += 1
        self.last_pruning_step = context.step
        
        return new_sparsity

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
