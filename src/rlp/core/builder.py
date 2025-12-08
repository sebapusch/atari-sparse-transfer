from gymnasium.vector import SyncVectorEnv
from omegaconf import DictConfig, OmegaConf

import torch.optim as optim
import torch
import gymnasium as gym

try:
    import wandb
except ImportError:
    wandb = None

from rlp.agent.base import AgentProtocol
from rlp.agent.ddqn import DDQNAgent
from rlp.agent.dqn import DQNAgent, DQNConfig
from rlp.components.encoders import NatureCNN, MinAtarCNN
from rlp.components.heads import LinearHead, DuelingHead
from rlp.components.network import QNetwork
from rlp.core.buffer import ReplayBuffer
from rlp.env.factory import make_env
from rlp.pruning.base import PrunerProtocol
from rlp.pruning.pruner import GMPPruner
from rlp.pruning.scheduler import LinearScheduler, CubicScheduler
from rlp.training.schedule import LinearSchedule, ScheduleProtocol
from rlp.core.trainer import TrainingConfig, TrainingContext
from rlp.core.checkpointer import Checkpointer
from rlp.core.logger import LoggerProtocol, WandbLogger, ConsoleLogger


class Builder:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = torch.device(self.config.device
                                   if (torch.cuda.is_available() and
                                       self.config.device == "cuda")
                                   else "cpu")

    def build_agent(self, num_actions: int, input_channels: int, pruner: PrunerProtocol | None) -> AgentProtocol:
        network = self.build_network(num_actions, input_channels)
        optimizer = optim.Adam(network.parameters(),
            lr=self.config.algorithm.optimizer.lr,
            eps=self.config.algorithm.optimizer.eps)

        dqn_config = DQNConfig(
            num_actions=num_actions,
            gamma=self.config.algorithm.gamma,
            tau=self.config.algorithm.tau,
            target_network_frequency=self.config.algorithm.target_network_frequency,
            prune_encoder_only=self.config.algorithm.get('prune_encoder_only', False)
        )

        params = {
            'network': network,
            'optimizer': optimizer,
            'pruner': pruner,
            'cfg': dqn_config,
            'device': self.device,
        }

        match self.config.algorithm.name:
            case 'dqn':
                agent = DQNAgent(**params)
            case 'ddqn':
                agent = DDQNAgent(**params)
            case _:
                raise ValueError(f"Unknown algorithm: {self.config.algorithm.name}")

        return agent

    def build_network(self, num_actions: int, input_channels: int) -> QNetwork:
        match self.config.algorithm.network.encoder:
            case 'nature_cnn':
                encoder = NatureCNN(input_channels=input_channels)
            case 'minatar_cnn':
                encoder = MinAtarCNN(input_channels=input_channels)
            case _:
                raise ValueError(f"Unknown encoder: {self.config.algorithm.network.encoder}")

        match self.config.algorithm.network.head:
            case 'linear':
                head = LinearHead(encoder.output_dim,
                                  num_actions,
                                  self.config.algorithm.network.hidden_dim)
            case 'dueling':
                head = DuelingHead(encoder.output_dim,
                                   num_actions,
                                   self.config.algorithm.network.hidden_dim)
            case _:
                raise ValueError(f'Unknown head: {self.config.algorithm.network.encoder}')

        return QNetwork(encoder, head)

    def build_pruner(self) -> PrunerProtocol | None:
        match self.config.pruning.method:
            case 'none':
                pruner = None
            case 'gmp':
                end_step = self.config.pruning.end_step \
                    if self.config.pruning.end_step is not None \
                    else self.config.train.total_timesteps

                match self.config.scheduler:
                    case 'linear':
                        scheduler = LinearScheduler(
                            self.config.pruning.initial_sparsity,
                            self.config.pruning.final_sparsity,
                            self.config.pruning.start_step,
                            end_step,
                        )
                    case 'cubic':
                        scheduler = CubicScheduler(
                            self.config.pruning.initial_sparsity,
                            self.config.pruning.final_sparsity,
                            self.config.pruning.start_step,
                            end_step,
                        )
                    case _:
                        raise ValueError(f"Unknown scheduler: {self.config.pruning.scheduler}")

                pruner = GMPPruner(
                    scheduler=scheduler,
                    update_frequency=self.config.pruning.update_frequency,
                )
            case 'lth':
                raise NotImplementedError('LTH pruner not implemented')
            case _:
                raise ValueError(f"Unknown pruner: {self.config.pruning.methed}")

        return pruner

    def build_envs(self) -> SyncVectorEnv:
        envs = []
        num_envs = self.config.env.num_envs

        if self.config.env.num_envs != 1:
            raise NotImplementedError(f"Only 1 environment is supported ({num_envs})")

        for i in range(num_envs):
            envs.append(make_env(
                self.config.env.id,
                self.config.seed + i,
                i,
                self.config.env.capture_video,
                self.config.wandb.name or "run"
            ))

        return gym.vector.SyncVectorEnv(envs)

    def build_replay_buffer(self, envs: SyncVectorEnv) -> ReplayBuffer:
        return ReplayBuffer(
            buffer_size=self.config.train.buffer_size,
            obs_dtype=envs.single_observation_space.dtype,
            obs_shape=envs.single_observation_space.shape,
            action_dtype=envs.single_action_space.dtype,
            action_shape=envs.single_action_space.shape,
            n_envs=envs.num_envs,
            optimize_memory_usage=self.config.env.get('frame_stack', 1) > 1,
            device=self.device,
        )

    def build_training_config(self) -> TrainingConfig:
        return TrainingConfig(
            learning_starts=self.config.algorithm.learning_starts,
            total_steps=self.config.train.total_timesteps,
            train_frequency=self.config.algorithm.train_frequency,
            save_frequency=self.config.train.checkpoint_interval,
            batch_size=self.config.algorithm.batch_size,
            seed=self.config.seed,
        )
    
    def build_epsilon_schedule(self) -> ScheduleProtocol:
        return LinearSchedule(
            start=self.config.algorithm.epsilon.start,
            end=self.config.algorithm.epsilon.end,
            duration=int(self.config.algorithm.epsilon.decay_fraction * self.config.train.total_timesteps)
        )

    def build_checkpointer(self) -> Checkpointer:
        return Checkpointer(
            checkpoint_base_dir=self.config.output_dir,
            run_id=self.config.wandb.get("id") # Handled by checkpointer/wandb logic if None
        )

    def build_logger(self) -> LoggerProtocol:
        if self.config.wandb.enabled:
            # Check for wandb import
            if wandb is None:
                print("Warning: wandb not installed, falling back to console logger")
                return ConsoleLogger()

            run_id = self.config.wandb.get("id")
            
            run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                group=self.config.wandb.group,
                tags=self.config.wandb.tags,
                job_type=self.config.wandb.job_type,
                name=self.config.wandb.name,
                config=OmegaConf.to_container(self.config, resolve=True),
                reinit=True,
                id=run_id,
                resume="allow" if run_id else None
            )
            return WandbLogger(run)
        else:
            return ConsoleLogger()
    
    def build_training_context(self, logger: LoggerProtocol) -> TrainingContext:
        envs = self.build_envs()
        num_actions = envs.single_action_space.n
        input_channels = envs.single_observation_space.shape[0]
        
        pruner = self.build_pruner()
        agent = self.build_agent(num_actions, input_channels, pruner=pruner)
        
        return TrainingContext(
            agent=agent,
            buffer=self.build_replay_buffer(envs),
            device=self.device,
            envs=envs,
            logger=logger,
            epsilon_scheduler=self.build_epsilon_schedule()
        )