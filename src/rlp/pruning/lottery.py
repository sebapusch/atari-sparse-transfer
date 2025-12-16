from __future__ import annotations

import copy
import json
import os
import torch
import torch.nn.utils.prune as prune
from dataclasses import dataclass, asdict

from rlp.core.trainer import Trainer
from rlp.pruning.utils import get_prunable_modules, calculate_sparsity

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class LotteryConfig:
    final_sparsity: float
    num_rounds: int
    rewind_to_step: int = 0
    description: str = "Lottery Ticket Hypothesis Experiment"


class Lottery:
    """
    Implements the Lottery Ticket Hypothesis pipeline (Frankle & Carbin, 2019).
    Wraps a Trainer instance to perform Iterative Magnitude Pruning (IMP) with rewinding.
    """

    def __init__(self, trainer: Trainer, config: LotteryConfig, resume_state: dict | None = None):
        self.trainer = trainer
        self.config = config

        # Calculate Initial Sparsity (e.g., from Transfer Learning)
        if hasattr(self.trainer.ctx.agent, "network") and hasattr(self.trainer.ctx.agent.network, "encoder"):
             self.initial_sparsity = calculate_sparsity(self.trainer.ctx.agent.network.encoder)
             print(f"Lottery: Detected initial sparsity of {self.initial_sparsity:.4f}")
        else:
             self.initial_sparsity = 0.0

        # Calculate schedule
        # Relation: (1 - final) = (1 - initial) * (retention ^ rounds)
        # retention ^ rounds = (1 - final) / (1 - initial)
        # retention = ((1 - final) / (1 - initial)) ** (1/rounds)
        
        if self.initial_sparsity >= self.config.final_sparsity:
            print("⚠️ Initial sparsity is already greater than or equal to final sparsity. Lottery will not prune.")
            self.retention_per_round = 1.0
        else:
            remaining_fraction = (1.0 - self.config.final_sparsity) / (1.0 - self.initial_sparsity)
            self.retention_per_round = remaining_fraction ** (1.0 / self.config.num_rounds)
            
        self.prune_amount = 1.0 - self.retention_per_round

        self.start_round = 1
        
        # Handle Resuming
        if resume_state and "lottery_round" in resume_state:
             self.start_round = resume_state["lottery_round"]

             print(f"Lottery: Resuming from Round {self.start_round}")
             
             # If resuming, we also need to load theta0
             self.rewind_state, self.rewind_opt_state = self._load_theta_0()
             
             if not self.rewind_state:
                 # Critical error? Or fallback to current state (incorrect for LTH)?
                 # Actually if resume_state exists, we MUST have theta_0 somewhere.
                 # Unless user deleted files.
                 raise RuntimeError("Resuming LTH but theta_0 not found.")
                 
        else:
            # Fresh Start
            self.rewind_state = copy.deepcopy(self.trainer.ctx.agent.state_dict())
            self.rewind_opt_state = copy.deepcopy(self.trainer.ctx.agent.optimizer.state_dict())
            self._save_theta_0()

    def run(self):
        """Executes the full LTH pipeline."""
        print(f"🎰 Starting Lottery Ticket Hypothesis Experiment")
        print(f"Initial Sparsity: {self.initial_sparsity:.4f}")
        print(f"Target Sparsity: {self.config.final_sparsity:.4f} over {self.config.num_rounds} rounds.")
        print(f"Retention rate per round: {self.retention_per_round:.4f}")
        print(f"Pruning rate (of remaining) per round: {self.prune_amount:.4f}")

        # If resuming, we might be in middle of round.
        # trainer.start_step tells us where we are in training loop.
        
        for round_idx in range(self.start_round, self.config.num_rounds + 1):
            print(f"\n--- 🔄 Round {round_idx}/{self.config.num_rounds} ---")
            
            # Update external state for checkpointing
            self.trainer.external_state["lottery_round"] = round_idx
            
            # 1. Train
            print(f"🏋️ Training Round {round_idx}...")
            # We continue training from wherever we are. 
            # If start of round, we might have just rewound (or initial start).
            self.trainer.train()
            
            # 2. Prune
            print(f"✂️ Pruning Network (Prune Rate: {self.prune_amount:.4f} of remaining)...")
            self._prune_network(amount=self.prune_amount)
            
            # 3. Rewind
            if round_idx < self.config.num_rounds:
                 print(f"⏪ Rewinding to theta_0 (Step {self.config.rewind_to_step})...")
                 self._rewind_network()
                 
                 # Reset trainer steps for next round re-training
                 # We want to re-train for the SAME duration or until convergence?
                 # Standard LTH: Train for T steps, Prune, Rewind to 0, Train for T steps...
                 # So we need to reset the trainer global step to 0 (or rewind_step)
                 # self.trainer.start_step = self.config.rewind_to_step 
                 # BUT Trainer doesn't reset automatically. We need to handle this.
                 
                 # Force Trainer reset
                 self.trainer.start_step = self.config.rewind_to_step
                 
                 # We also need to reset scheduler / etc if they depend on step?
                 # Trainer.train() loop uses self.start_step. 
                 
            # 4. Log / Measurement
            if hasattr(self.trainer.ctx.agent, "network"):
                 # Fallback if no specific encoder attribute
                 # But we pruned 'encoder', so it SHOULD exist.
                 current_sparsity = calculate_sparsity(self.trainer.ctx.agent.network)
            else:
                 current_sparsity = 0.0

            print(f"📊 Round {round_idx} complete. Current Sparsity (Encoder): {current_sparsity:.4f}")
            
            self._log_artifacts(round_idx, current_sparsity)

        print("\n🏆 Lottery Ticket Experiment Complete.")
        
    def _save_theta_0(self):
        """Persists init weights to disk/WandB to allow resuming."""
        path = os.path.join(self.trainer.checkpointer.checkpoint_dir, "theta_0.pt")
        torch.save({
            "agent": self.rewind_state,
            "optimizer": self.rewind_opt_state
        }, path)
        print(f"Saved theta_0 to {path}")

    def _load_theta_0(self):
        path = os.path.join(self.trainer.checkpointer.checkpoint_dir, "theta_0.pt")
        if os.path.exists(path):
            data = torch.load(path, map_location=self.trainer.ctx.device)
            print(f"Loaded theta_0 from {path}")
            return data["agent"], data["optimizer"]
        return None, None

    def _prune_network(self, amount: float):
        """
        Applies global magnitude pruning to the agent's network encoder.
        Prunes to reach 'amount' total sparsity.
        """
        if not hasattr(self.trainer.ctx.agent, "network"):
            raise RuntimeError("Agent must have a 'network' attribute to be pruned.")
        
        # Prune only the encoder
        network = self.trainer.ctx.agent.network.encoder
        
        parameters_to_prune = get_prunable_modules(network)
        
        if not parameters_to_prune:
            print("⚠️ No prunable parameters found.")
            return

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        
    def _rewind_network(self):
        """
        Resets unpruned weights to their original values (theta_0).
        Resets optimizer state.
        """
        agent = self.trainer.ctx.agent
        
        # Rewind weights
        orig_net_state = self.rewind_state['network']
        
        for name, module in agent.network.named_modules():
            # If pruned, we have weight_orig/mask. Reset weight_orig.
            if prune.is_pruned(module):
                weight_key = f"{name}.weight" if name else "weight"
                if weight_key in orig_net_state:
                    original_weight = orig_net_state[weight_key]
                    module.weight_orig.data.copy_(original_weight.data)
            else:
                 # Standard rewinding for unpruned modules
                 weight_key = f"{name}.weight" if name else "weight"
                 if hasattr(module, 'weight') and weight_key in orig_net_state:
                     module.weight.data.copy_(orig_net_state[weight_key].data)
                 
                 bias_key = f"{name}.bias" if name else "bias"
                 if hasattr(module, 'bias') and module.bias is not None and bias_key in orig_net_state:
                     module.bias.data.copy_(orig_net_state[bias_key].data)

        # Reset Optimizer
        agent.optimizer.load_state_dict(self.rewind_opt_state)
        
        # Reset Target Network
        agent.target_network.load_state_dict(self.rewind_state['target_network'])

    def _log_artifacts(self, round_idx: int, sparsity: float):
        # ... (start of method)
        
        # Save Mask
        # We extract just the masks.
        masks = {}
        # Iterate over full network to capture all masks (which should only be in encoder now)
        for name, module in self.trainer.ctx.agent.network.named_modules():
            if prune.is_pruned(module):
                masks[name] = module.weight_mask.cpu() # Keep on CPU
        
        artifact_name = f"lottery_ticket_round_{round_idx}"
        artifact = wandb.Artifact(artifact_name, type="model")
        
        # Save to tmp file
        save_dir = f"artifacts/round_{round_idx}"
        os.makedirs(save_dir, exist_ok=True)
        
        mask_path = os.path.join(save_dir, "mask.pt")
        torch.save(masks, mask_path)
        artifact.add_file(mask_path)
        
        # Save Metadata
        meta_path = os.path.join(save_dir, "lottery_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        artifact.add_file(meta_path)
        
        # Save Rewound Weights (Optimization: only save if needed, but per-req we should)
        # "rewound initialization values for surviving weights"
        # We can just save the current agent state dict, as it IS the rewound state right now.
        rec_path = os.path.join(save_dir, "init_weights.pt")
        torch.save(self.trainer.ctx.agent.state_dict(), rec_path)
        artifact.add_file(rec_path)

        self.trainer.ctx.logger.run.log_artifact(artifact)
        
        # Clean up
        # shutil.rmtree(save_dir) # Optional cleanup
