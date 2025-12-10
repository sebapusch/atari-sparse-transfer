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
    rewind_to_step: int = 0  # Should be 0 for standard LTH
    description: str = "Lottery Ticket Hypothesis Experiment"


class Lottery:
    """
    Implements the Lottery Ticket Hypothesis pipeline (Frankle & Carbin, 2019).
    Wraps a Trainer instance to perform Iterative Magnitude Pruning (IMP) with rewinding.
    """

    def __init__(self, trainer: Trainer, config: LotteryConfig, resume_state: dict | None = None):
        self.trainer = trainer
        self.config = config
        
        # Calculate per-round pruning rate to reach final_sparsity in num_rounds
        self.retention_per_round = (1.0 - self.config.final_sparsity) ** (1.0 / self.config.num_rounds)
        self.prune_amount = 1.0 - self.retention_per_round

        self.start_round = 1
        
        # Handle Resuming
        if resume_state and "lottery_round" in resume_state:
             # Resume logic
             self.start_round = resume_state["lottery_round"]
             # If we finished logic for round X, we saved lottery_round=X.
             # So we should start next round?
             # Wait, usually we save periodically. 
             # If we resume from middle of training in round X, 'lottery_round' might be X.
             # In that case, we should continue round X.
             # If we finished round X and saved, then crashed, we might be at start of X+1.
             # Let's assume 'lottery_round' indicates the round currently IN PROGRESS or just FINISHED if we call update at end?
             # We will update 'lottery_round' at START of round.
             print(f"🔄 Lottery: Resuming from Round {self.start_round}")
             
             # Attempt to load theta_0 from local disk (persisted from previous run)
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
        print(f"Target Sparsity: {self.config.final_sparsity:.4f} over {self.config.num_rounds} rounds.")
        print(f"Pruning rate per round: {self.prune_amount:.4f}")

        # If resuming, we might be in middle of round.
        # trainer.start_step tells us where we are in training loop.
        
        for round_idx in range(self.start_round, self.config.num_rounds + 1):
            print(f"\n--- 🔄 Round {round_idx}/{self.config.num_rounds} ---")
            
            # Update external state for checkpointing
            self.trainer.external_state['lottery_round'] = round_idx
            
            # 1. Train to completion
            # If resuming round_idx (== start_round) and trainer.start_step > 0, 
            # we just continue training.
            if round_idx > self.start_round:
                 # Only reset if we are NOT resuming this specific round from middle
                 self.trainer.ctx.envs.reset(seed=self.trainer.cfg.seed + round_idx)
                 self.trainer.ctx.buffer.reset()
                 self.trainer.start_step = 0 
            
            # Train
            self.trainer.train()

            # 2. Compute Global Magnitude Mask & 3. Apply Mask
            print(f"✂️ Pruning round {round_idx}...")
            self._prune_network()

            # 4. Rewind surviving weights
            print(f"⏪ Rewinding weights to init...")
            self._rewind_network()

            # 5. Sync/Log
            # 5. Sync/Log
            # current_sparsity = calculate_sparsity(self.trainer.ctx.agent.network.encoder) # REMOVED UNSAFE CALL

            
            # We assume agent.network is the module to prune.
            if hasattr(self.trainer.ctx.agent, "network"):
                network = self.trainer.ctx.agent.network
                current_sparsity = calculate_sparsity(network)
            else:
                # Fallback, though unsafe
                current_sparsity = 0.0
                print("⚠️ Could not calculate sparsity: agent has no 'network' attribute.")

            print(f"📊 Round {round_idx} complete. Current Sparsity: {current_sparsity:.4f}")
            
            self._log_artifacts(round_idx, current_sparsity)

        print("\n🏆 Lottery Ticket Experiment Complete.")
        
    def _save_theta_0(self):
        """Persists init weights to disk/WandB to allow resuming."""
        # We save locally to checkpoint dir so it's available on resume
        # and checking 'latest' checkpoint dir is handled by checkpointer logic?
        # Maybe we should ask Checkpointer for dir?
        # self.trainer.checkpointer.checkpoint_dir
        
        path = os.path.join(self.trainer.checkpointer.checkpoint_dir, "theta_0.pt")
        torch.save({
            "agent": self.rewind_state,
            "optimizer": self.rewind_opt_state
        }, path)
        print(f"💾 Saved theta_0 to {path}")

    def _load_theta_0(self):
        path = os.path.join(self.trainer.checkpointer.checkpoint_dir, "theta_0.pt")
        if os.path.exists(path):
            data = torch.load(path, map_location=self.trainer.ctx.device)
            print(f"📂 Loaded theta_0 from {path}")
            return data["agent"], data["optimizer"]
        
        # Try checking if remote WandB has it? 
        # For now assume local filesystem persistence or user manually downloaded.
        return None, None
        
    def _prune_network(self):
        """
        Applies global magnitude pruning to the agent's network.
        Prunes self.prune_amount of the *remaining* weights.
        """
        if not hasattr(self.trainer.ctx.agent, "network"):
            raise RuntimeError("Agent must have a 'network' attribute to be pruned.")
        
        network = self.trainer.ctx.agent.network
        
        # Get prunable modules (Linear, Conv2d)
        # Using utils.get_prunable_modules which returns list of (module, name)
        parameters_to_prune = get_prunable_modules(network)
        
        if not parameters_to_prune:
            print("⚠️ No prunable parameters found.")
            return

        # Global LS Unstructured pruning
        # This applies the mask permanently to the module state (creates _mask and _orig)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.prune_amount,
        )
        
        # Note: prune.global_unstructured applies the mask immediately.
        # But for LTH we want to prune THEN rewind.
        
    def _rewind_network(self):
        """
        Resets unpruned weights to their original values (theta_0).
        Resets optimizer state.
        """
        agent = self.trainer.ctx.agent
        
        # 1. Reload original weights
        # This will separate mask and value if prune was applied correctly.
        # When we load_state_dict on a pruned model, PyTorch handles it if keys match.
        # BUT: The saved state_dict (theta_0) has 'weight', not 'weight_orig' + 'weight_mask'.
        # The current model has 'weight_orig' and 'weight_mask' due to pruning.
        
        # Complex Rewinding Logic:
        # We must copy theta_0 values into 'weight_orig' of the current model.
        # We must NOT overwrite the masks.
        
        # Iterate over modules and manualy reset 'weight_orig'
        current_modules = dict(agent.network.named_modules())
        
        # We need the original state dict to look up values.
        # The keys in rewind_state are like 'network.encoder.0.weight'
        # or just 'encoder.0.weight' depending on agent.state_dict() structure.
        # DQNAgent.state_dict() returns {'network': ..., ...}
        
        orig_net_state = self.rewind_state['network']
        
        for name, module in agent.network.named_modules():
            # We are looking for pruned modules
            if prune.is_pruned(module):
                # This module has 'weight_orig' and 'weight_mask'
                # We want to reset 'weight_orig' to the value from theta_0.
                
                # Construct key for lookup in orig_net_state
                # name is relative to network. e.g. "encoder.0"
                # We need to find the weight param.
                # Usually it's "weight".
                
                weight_key = f"{name}.weight" if name else "weight"
                
                if weight_key in orig_net_state:
                    # Update weight_orig
                    # Ensure shapes match
                    original_weight = orig_net_state[weight_key]
                    module.weight_orig.data.copy_(original_weight.data)
                else:
                    # Maybe bias? Pruning usually just weights.
                    pass
            else:
                 # Not pruned, but we still rewind it! LTH rewinds ALL weights.
                 # Just standard load_state_dict for these would work, but mixing is tricky.
                 # Let's handle manually to be safe.
                 
                 # Weights
                 weight_key = f"{name}.weight" if name else "weight"
                 if hasattr(module, 'weight') and weight_key in orig_net_state:
                     module.weight.data.copy_(orig_net_state[weight_key].data)
                 
                 # Biases
                 bias_key = f"{name}.bias" if name else "bias"
                 if hasattr(module, 'bias') and module.bias is not None and bias_key in orig_net_state:
                     module.bias.data.copy_(orig_net_state[bias_key].data)

        # 2. Reset Optimizer
        # We load the original optimizer state.
        # Note: Functionally this resets momentum buffers etc.
        agent.optimizer.load_state_dict(self.rewind_opt_state)
        
        # 3. Handle Target Network
        # In DQNAgent, target network is a separate container.
        # We should reset it too to match theta_0.
        # But wait, does it have masks?
        # The agent.prune() logic in DQNAgent assumes clean target network.
        # But if we are running LTH, does the target network need to be pruned?
        # Usually LTH implies the architecture itself is pruned.
        # The DQNAgent handles masking via `_update_target_network` which COPIES masked weights to target.
        # So target network structure (dense) remains, but values effectively become sparse.
        # So we just need to rewind target network to theta_0 target network.
        
        agent.target_network.load_state_dict(self.rewind_state['target_network'])

    def _log_artifacts(self, round_idx: int, sparsity: float):
        if wandb is None or self.trainer.ctx.logger.run is None:
            return

        # Prepare metadata
        metadata = {
            "round": round_idx,
            "sparsity": sparsity,
            "method": "LTH_Global_Magnitude",
            "config": asdict(self.config)
        }
        
        # Save Mask
        # We extract just the masks.
        masks = {}
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
