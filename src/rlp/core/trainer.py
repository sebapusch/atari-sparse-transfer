from __future__ import annotations

import time
import torch
import torch.optim as optim
import numpy as np
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf

from rlp.core.logger import LoggerProtocol
from rlp.core.buffer import ReplayBuffer
from rlp.agent.base import AgentProtocol
from rlp.agent.dqn import DQNAgent
from rlp.components.network import QNetwork
from rlp.components.encoders import NatureCNN, MinAtarCNN
from rlp.components.heads import LinearHead, DuelingHead
from rlp.pruning.base import PrunerProtocol
from rlp.pruning.pruner import GMPPruner, RandomPruner, BasePruner
from rlp.pruning.scheduler import ConstantScheduler, LinearScheduler, CubicScheduler
from rlp.env.factory import make_env

class Trainer:
    def __init__(self, cfg: DictConfig, logger: LoggerProtocol) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
        
        # Initialize Environment
        # Vectorized env
        import gymnasium as gym
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(cfg.env.id, cfg.seed + i, i, cfg.env.capture_video, cfg.wandb.name or "run") 
             for i in range(cfg.env.num_envs)]
        )
        
        # Initialize Network
        if cfg.algorithm.network.encoder == "nature_cnn":
            encoder = NatureCNN(input_channels=4 if cfg.env.grayscale else 3) # Stack=4 usually
        elif cfg.algorithm.network.encoder == "minatar_cnn":
            encoder = MinAtarCNN(input_channels=self.envs.single_observation_space.shape[0])
        else:
            raise ValueError(f"Unknown encoder: {cfg.algorithm.network.encoder}")
            
        if cfg.algorithm.network.head == "linear":
            head = LinearHead(encoder.output_dim, self.envs.single_action_space.n, cfg.algorithm.network.hidden_dim)
        elif cfg.algorithm.network.head == "dueling":
            head = DuelingHead(encoder.output_dim, self.envs.single_action_space.n, cfg.algorithm.network.hidden_dim)
        else:
            raise ValueError(f"Unknown head: {cfg.algorithm.network.head}")
            
        network = QNetwork(encoder, head)
        
        # Initialize Optimizer
        optimizer = optim.Adam(network.parameters(), lr=cfg.algorithm.optimizer.lr, eps=cfg.algorithm.optimizer.eps)
        
        # Initialize Agent
        if cfg.algorithm.name == "dqn":
            self.agent = DQNAgent(
                network=network,
                optimizer=optimizer,
                num_actions=self.envs.single_action_space.n,
                gamma=cfg.algorithm.gamma,
                tau=cfg.algorithm.tau,
                target_network_frequency=cfg.algorithm.target_network_frequency,
                double_dqn=False,
                device=str(self.device)
            )
        elif cfg.algorithm.name == "ddqn":
            self.agent = DQNAgent(
                network=network,
                optimizer=optimizer,
                num_actions=self.envs.single_action_space.n,
                gamma=cfg.algorithm.gamma,
                tau=cfg.algorithm.tau,
                target_network_frequency=cfg.algorithm.target_network_frequency,
                double_dqn=True,
                device=str(self.device)
            )
        else:
            raise ValueError(f"Unknown algorithm: {cfg.algorithm.name}")

        # Initialize Pruner
        if cfg.pruning.method == "none":
            scheduler = ConstantScheduler(0.0)
            self.pruner = BasePruner(scheduler)
        elif cfg.pruning.method == "gmp":
            # Example scheduler config mapping
            # Assuming cfg.pruning.schedule exists
            scheduler = LinearScheduler(0.0, 0.9, 0, cfg.train.total_timesteps // 2) # Placeholder
            self.pruner = GMPPruner(scheduler)
        else:
            # Fallback
            scheduler = ConstantScheduler(0.0)
            self.pruner = BasePruner(scheduler)

        # Initialize Buffer
        self.buffer = ReplayBuffer(
            cfg.train.buffer_size,
            self.envs.single_observation_space.shape,
            self.envs.single_action_space.shape,
            self.device,
            n_envs=cfg.env.num_envs,
            obs_dtype=self.envs.single_observation_space.dtype,
            action_dtype=self.envs.single_action_space.dtype,
        )

    def train(self) -> None:
        start_time = time.time()
        obs, _ = self.envs.reset(seed=self.cfg.seed)
        
        for global_step in range(self.cfg.train.total_timesteps):
            # Epsilon schedule
            epsilon = self.linear_schedule(
                self.cfg.algorithm.epsilon.start,
                self.cfg.algorithm.epsilon.end,
                self.cfg.algorithm.epsilon.decay_fraction * self.cfg.train.total_timesteps,
                global_step
            )
            
            # Action selection
            if global_step < self.cfg.algorithm.learning_starts:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                actions = self.agent.get_action(obs, epsilon=epsilon)
                
            # Step
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
            
            # Record rewards
            # Record rewards
            if "episode" in infos:
                # Gymnasium >= 1.0 vector env logging
                # infos['episode'] is a dict of arrays, with '_episode' (or '_r') as mask
                env_mask = infos.get("_episode", infos.get("_r"))
                if env_mask is not None:
                    for i in range(self.envs.num_envs):
                        if env_mask[i]:
                            self.logger.log_metrics({
                                "charts/episodic_return": infos["episode"]["r"][i],
                                "charts/episodic_length": infos["episode"]["l"][i],
                                "charts/epsilon": epsilon,
                            }, step=global_step)
                            print(f"Step: {global_step}, Return: {infos['episode']['r'][i]}")
            elif "final_info" in infos:
                # Legacy Gymnasium logging
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        self.logger.log_metrics({
                            "charts/episodic_return": info["episode"]["r"],
                            "charts/episodic_length": info["episode"]["l"],
                            "charts/epsilon": epsilon,
                        }, step=global_step)
                        print(f"Step: {global_step}, Return: {info['episode']['r']}")

            # Buffer add
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)
            
            obs = next_obs
            
            # Training
            if global_step > self.cfg.algorithm.learning_starts:
                if global_step % self.cfg.algorithm.train_frequency == 0:
                    data = self.buffer.sample(self.cfg.algorithm.batch_size)
                    metrics = self.agent.update(data)
                    
                    if global_step % self.cfg.train.log_interval == 0:
                        metrics["charts/SPS"] = int(global_step / (time.time() - start_time))
                        self.logger.log_metrics(metrics, step=global_step)
                        
                    # Pruning update
                    pruning_metrics = self.pruner.update(self.agent.network, global_step)
                    if pruning_metrics:
                        self.logger.log_metrics(pruning_metrics, step=global_step)
                        
                # Pruning on_step
                self.pruner.on_step(self.agent.network, global_step)

        self.envs.close()
        self.logger.close()

    @staticmethod
    def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)
