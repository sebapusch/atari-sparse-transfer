from __future__ import annotations

import time
import os
from array import ArrayType
from dataclasses import dataclass

import torch
import torch.nn.utils.prune as prune
import numpy as np
import gymnasium as gym

from rlp.core.checkpointer import Checkpointer
from rlp.core.logger import LoggerProtocol
from rlp.core.buffer import ReplayBuffer
from rlp.agent.base import AgentProtocol
from rlp.pruning.base import PrunerProtocol
from rlp.training.schedule import ScheduleProtocol

@dataclass
class TrainingContext:
    agent: AgentProtocol
    buffer: ReplayBuffer
    device: torch.device
    envs: gym.vector.SyncVectorEnv
    logger: LoggerProtocol
    epsilon_scheduler: ScheduleProtocol

@dataclass
class TrainingConfig:
    learning_starts: int
    total_steps: int
    train_frequency: int
    save_frequency: int
    batch_size: int
    seed: int


class Trainer:
    def __init__(self, ctx: TrainingContext, cfg: TrainingConfig, checkpointer: Checkpointer) -> None:
        self.ctx = ctx
        self.cfg = cfg
        self.checkpointer = checkpointer

        self.start_step = 0

    def train(self) -> None:
        envs = self.ctx.envs
        obs, _ = envs.reset(seed=self.cfg.seed)

        global_step = self.start_step
        
        while global_step <= self.cfg.total_steps:
            epsilon = self.ctx.epsilon_scheduler[global_step]

            actions = self._get_actions(obs, global_step, epsilon)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
            self._log_episodic_metrics(global_step, infos, epsilon)

            self._update_buffer(
                obs,
                next_obs,
                actions,
                terminations,
                rewards,
                infos,
                truncations
            )

            obs = next_obs

            if not self._is_training_step(global_step):
                global_step += 1
                continue

            ### Training
            metrics = {}
            batch = self.ctx.buffer.sample(self.cfg.batch_size)

            for name, value in self.ctx.agent.update(batch):
                metrics[f"charts/{name}"] = value

            sparsity = self.ctx.agent.prune(global_step)

            if sparsity is not None:
                metrics["charts/sparsity"] = sparsity

            self.ctx.logger.log_metrics(metrics, step=global_step)

            if self.cfg.save_frequency % global_step == 0:
                self._save(global_step, epsilon)

            global_step += 1

        self._save(step=global_step, epsilon=0.0)
        self.ctx.envs.close()
        self.ctx.logger.close()

    def _is_training_step(self, step: int) -> bool:
        return (step >= self.cfg.learning_starts and
                step % self.cfg.train_frequency == 0)

    def _save(self, step: int, epsilon: float) -> None:
        self.checkpointer.save(
            step,
            state={
                'agent': self.ctx.agent.state_dict(),
                'cfg':   self.cfg,
                'step':  step,
            },
            metadata={
                'epsilon': epsilon,
            }
        )

    def _try_resume(self):
        """
        Loads the latest checkpoint if available and restores state.
        """
        state = self.checkpointer.load(self.ctx.device)

        if state is None:
            print("🆕 Starting training from scratch.")
            return

        print(f"🔄 Resuming training from step {state['step']}...")
        self.ctx.agent.load_state_dict(state['agent'])
        self.cfg = TrainingConfig(**state['cfg'])
        self.start_step = state['step'] + 1

    def _get_actions(self, obs: np.ndarray, step: int, epsilon: float) -> np.ndarray:
        if step < self.cfg.learning_starts:
            return np.array([
                self.ctx.envs.single_action_space.sample()
                for _ in range(self.ctx.envs.num_envs)
            ])

        return self.ctx.agent.select_action(obs, epsilon=epsilon)

    def _log_episodic_metrics(self, step: int, infos: dict, epsilon: float) -> None:
        episodic_return = None
        episodic_length = None

        if "episode" in infos:
            # Gymnasium >= 1.0 vector env logging
            # infos['episode'] is a dict of arrays, with '_episode' (or '_r') as mask
            env_mask = infos.get("_episode", infos.get("_r"))
            if env_mask is not None:
                for i in range(self.ctx.envs.num_envs):
                    if env_mask[i]:
                        episodic_return = infos["episode"]["r"][i]
                        episodic_length = infos["episode"]["l"][i]
        elif "final_info" in infos:
            # Legacy Gymnasium logging
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_return = info["episode"]["r"]
                    episodic_length = info["episode"]["l"]

        if episodic_return is None:
            return

        self.ctx.logger.log_metrics({
            "charts/episodic_return": episodic_return,
            "charts/episodic_length": episodic_length,
            "charts/epsilon": epsilon,
        }, step)

    def _update_buffer(self,
                       obs: np.ndarray,
                       next_obs: np.ndarray,
                       actions: np.ndarray,
                       terminations: np.ndarray,
                       rewards: np.ndarray,
                       infos: dict,
                       truncations: ArrayType):
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        self.ctx.buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

    def save_checkpoint(self, step: int) -> None:
        """Save model checkpoint and optionally sparsity masks."""
        path = self.cfg.output_dir
        os.makedirs(path, exist_ok=True)
        
        # Determine prefix
        prefix = ""
        if self.cfg.wandb.name:
            prefix = f"{self.cfg.wandb.name}_"
        
        # Save model weights
        model_path = os.path.join(path, f"{prefix}model_{step}.pt")
        torch.save(self.agent.network.state_dict(), model_path)
        print(f"Saved model to {model_path}")
        
        # Save sparsity masks
        if self.cfg.train.get("save_sparsity_mask", False):
            masks = {}
            
            # Better approach for masks: iterate named_modules of the network
            for name, module in self.agent.network.named_modules():
                if prune.is_pruned(module):
                    for hook in module._forward_pre_hooks.values():
                        if isinstance(hook, prune.BasePruningMethod):
                            # Usually the mask is stored as buffer named "{parameter_name}_mask"
                            # The hook._tensor_name gives the parameter name (e.g. 'weight')
                            mask_name = f"{name}.{hook._tensor_name}_mask"
                            mask = getattr(module, hook._tensor_name + "_mask")
                            masks[mask_name] = mask.cpu()
            
            if masks:
                mask_path = os.path.join(path, f"{prefix}masks_{step}.pt")
                torch.save(masks, mask_path)
                print(f"Saved sparsity masks to {mask_path}")

