from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

from rlp.agent.base import AgentProtocol
from rlp.components.network import QNetwork
from rlp.core.buffer import ReplayBufferSamples
from rlp.pruning.base import PrunerProtocol


@dataclass
class DQNConfig:
    num_actions: int
    gamma: float
    target_network_frequency: int
    tau: float = 1.0
    prune_encoder_only: bool = False


class DQNAgent(AgentProtocol):
    def __init__(
        self,
        network: QNetwork,
        optimizer: optim.Optimizer,
        pruner: PrunerProtocol | None,
        cfg: DQNConfig,
        device: torch.Device,
    ) -> None:
        self.device = device
        self.network = network.to(self.device)
        self.optimizer = optimizer
        self.pruner = pruner
        self.cfg = cfg

        self.target_network = copy.deepcopy(network).to(self.device)

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        if obs.ndim == 3:
             obs = np.expand_dims(obs, axis=0)
        
        batch_size = obs.shape[0]

        if random.random() < epsilon:
            return np.random.randint(0, self.cfg.num_actions, size=batch_size)
        
        with torch.no_grad():
            q_values = self.network(torch.Tensor(obs).to(self.device))
            
            # Random tie-breaking:
            # Add small random noise to Q-values before argmax.
            # This ensures that if multiple actions have the same max Q-value,
            # the choice is randomized based on the seeded noise.
            noise = torch.rand_like(q_values) * 1e-5
            actions = torch.argmax(q_values + noise, dim=1).cpu().numpy()
        
        return actions

    def update(self, batch: ReplayBufferSamples, step: int) -> dict[str, float]:
        batch = batch.to(self.device)

        td_target = self._compute_td_target(batch.rewards, batch.next_observations, batch.dones)

        old_val = self.network(batch.observations).gather(1, batch.actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % self.cfg.target_network_frequency == 0:
            self.update_target_network()

        return {
            "loss": loss.item(),
            "q_values": old_val.mean().item(),
        }

    def finished_training(self, step: int) -> None:
        self.update_target_network()

    def prune(self, context: Any) -> float | None:
        """
        Context is typed as Any here to avoid circular dependencies with PruningContext 
        but implementation expects PruningContext.
        """
        if self.pruner is None:
            return None

        return self.pruner.prune(self.network.encoder, context)


    def should_stop(self, context: Any) -> bool:
        if self.pruner is None:
            return False
        return self.pruner.should_stop(context)

    def update_target_network(self) -> None:
        """
        Mask aware target network update.
        Assumes self.target_network is structurally identical to self.network
        BUT is not registered with pruning hooks (i.e., it is a "clean" container).
        """
        # 1. Get all buffers (needed for masks AND batchnorm stats)
        source_buffers = dict(self.network.named_buffers())
        target_buffers = dict(self.target_network.named_buffers())

        # 2. Update Parameters (Polyak Averaging)
        target_params = dict(self.target_network.named_parameters())

        for src_name, src_param in self.network.named_parameters():
            # Determine the effective name in the target network
            if src_name.endswith("_orig"):
                # Pruned layer: Reconstruct the effective weight
                mask_name = src_name.replace("_orig", "_mask")
                target_name = src_name.replace("_orig", "")

                if mask_name in source_buffers:
                    # W_eff = W_orig * Mask
                    source_data = src_param.data * source_buffers[mask_name]
                else:
                    # Fallback if mask is missing (unlikely if named _orig)
                    source_data = src_param.data
            else:
                # Unpruned layer
                target_name = src_name
                source_data = src_param.data

            # Safety check: Ensure target exists
            if target_name in target_params:
                # In-place Polyak averaging: param_target = tau * param_src + (1-tau) * param_target
                # torch.lerp is equivalent to: input + weight * (end - input)
                # We want: tau * src + (1-tau) * target
                # which matches lerp(target, src, tau)
                target_params[target_name].data.lerp_(source_data, self.cfg.tau)

        # 3. Update Buffers (Hard Copy for BatchNorm, etc.)
        # We iterate source buffers to copy running_mean/var, but we must skip the masks
        # because the target network (unpruned) doesn't need/have mask buffers.
        for name, buffer in source_buffers.items():
            if not name.endswith("_mask") and name in target_buffers:
                target_buffers[name].data.copy_(buffer.data)

    def _compute_td_target(self,
                           rewards: torch.Tensor,
                           next_obs: torch.Tensor,
                           dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_max, _ = self.target_network(next_obs).max(dim=1)

            return rewards + self.cfg.gamma * target_max * (1 - dones)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._handle_pruned_loading(self.network, state_dict["network"])
        self.network.load_state_dict(state_dict["network"])
        
        self._handle_pruned_loading(self.target_network, state_dict["target_network"])
        self.target_network.load_state_dict(state_dict["target_network"])
        
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def _handle_pruned_loading(self, module: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
        for key in state_dict.keys():
            if key.endswith("_mask"):
                # expected key format: path.to.module.param_mask
                # corresponding param: path.to.module.param
                param_name = key.split(".")[-1].replace("_mask", "")
                module_path = ".".join(key.split(".")[:-1])
                
                submodule = module
                if module_path:
                    # Traverse to submodule
                    try:
                        for part in module_path.split("."):
                            submodule = getattr(submodule, part)
                    except AttributeError:
                        # If structure doesn't match, we can't prune anyway. 
                        # Actual load_state_dict will raise RuntimeError about missing keys later.
                        continue
                
                # Check if already pruned.
                # If pruned, it should have {param_name}_orig
                if not hasattr(submodule, param_name + "_orig"):
                     prune.identity(submodule, param_name)

    def to(self, device: torch.device) -> DQNAgent:
        self.device = device
        self.network.to(device)
        self.target_network.to(device)
        return self
