from __future__ import annotations

from array import ArrayType
from dataclasses import dataclass

import torch
import numpy as np
import gymnasium as gym

from rlp.core.checkpointer import Checkpointer
from rlp.core.logger import LoggerProtocol
from rlp.core.buffer import ReplayBuffer
from rlp.agent.base import AgentProtocol
from rlp.training.schedule import ScheduleProtocol
from rlp.pruning.base import PruningContext

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
    log_interval: int
    batch_size: int
    seed: int
    delegate_stopping: bool = False


class Trainer:
    def __init__(self, ctx: TrainingContext, cfg: TrainingConfig, checkpointer: Checkpointer, start_step: int = 0) -> None:
        self.ctx = ctx
        self.cfg = cfg
        self.checkpointer = checkpointer
        self.external_state: dict[str, Any] = {}

    def train(self) -> None:
        envs = self.ctx.envs
        obs, _ = envs.reset(seed=self.cfg.seed)

        global_step = 0
        
        self.recent_returns = []
        
        while True:
            if self._should_stop(global_step):
                break

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

            batch = self.ctx.buffer.sample(self.cfg.batch_size)
            agent_metrics = self.ctx.agent.update(batch, step=global_step)

            if global_step % self.cfg.log_interval == 0:
                metrics = {}
                for metric in agent_metrics:
                    metrics[f"charts/{metric}"] = agent_metrics[metric]

                sparsity = self._run_pruning(global_step)

                if sparsity is not None:
                    metrics["pruning/sparsity"] = sparsity
                    # Clear recent returns to force new data collection before next convergence check
                    self.recent_returns = []

                self.ctx.logger.log_metrics(metrics, step=global_step)

            if self.cfg.save_frequency % global_step == 0:
                self._save(global_step, epsilon)

            global_step += 1

        self.ctx.agent.finished_training(global_step)
        self._save(step=global_step, epsilon=0.0)
        self.ctx.envs.close()
        self.ctx.logger.close()

    def _should_stop(self, step: int) -> bool:
        if self.cfg.delegate_stopping:
            return self.ctx.agent.should_stop(self._get_pruning_context(step))
        return step > self.cfg.total_steps

    def _get_pruning_context(self, step: int) -> PruningContext:
        return PruningContext(
            step=step,
            agent=self.ctx.agent,
            recent_episodic_returns=self.recent_returns
        )

    def _run_pruning(self, step: int) -> float | None:
         ctx = self._get_pruning_context(step)
         return self.ctx.agent.prune(ctx)

    def _is_training_step(self, step: int) -> bool:
        return (step >= self.cfg.learning_starts and
                step % self.cfg.train_frequency == 0)

    def _save(self, step: int, epsilon: float) -> None:
        state = {
            'agent': self.ctx.agent.state_dict(),
            'cfg':   self.cfg,
            'step':  step,
        }
        
        # Merge external state if present
        if self.external_state:
            state.update(self.external_state)
            
        self.checkpointer.save(
            step,
            state=state,
            metadata={
                'epsilon': epsilon,
            }
        )
        self.ctx.logger.commit()

    def _get_actions(self, obs: np.ndarray, step: int, epsilon: float) -> np.ndarray:
        if step < self.cfg.learning_starts:
            return np.array([
                self.ctx.envs.single_action_space.sample()
                for _ in range(self.ctx.envs.num_envs)
            ])

        return self.ctx.agent.select_action(obs, epsilon=epsilon)

    def _log_episodic_metrics(self, step: int, infos: dict, epsilon: float) -> None:
        returns = []
        lengths = []

        if "episode" in infos:
            # Vectorized efficient check
            env_mask = infos.get("_episode", infos.get("_r"))
            if env_mask is not None:
                returns.extend(infos["episode"]["r"][env_mask])
                lengths.extend(infos["episode"]["l"][env_mask])

        elif "final_info" in infos:
            # Legacy loop
            for info in infos["final_info"]:
                if info and "episode" in info:
                    returns.append(info["episode"]["r"])
                    lengths.append(info["episode"]["l"])

        if not returns:
            return

        self.recent_returns.extend(returns)
        # Keep last 100
        if len(self.recent_returns) > 100:
            self.recent_returns = self.recent_returns[-100:]

        # Log mean to capture trend of the batch
        self.ctx.logger.log_metrics({
            "charts/episodic_return": np.mean(returns),
            "charts/episodic_length": np.mean(lengths),
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

