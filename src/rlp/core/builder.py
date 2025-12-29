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
from rlp.pruning.lottery import LotteryPruner, LotteryConfig
from rlp.pruning.scheduler import LinearScheduler, CubicScheduler
from rlp.training.schedule import LinearSchedule, ScheduleProtocol
from rlp.core.trainer import TrainingConfig, TrainingContext
from rlp.core.checkpointer import Checkpointer

from rlp.core.logger import LoggerProtocol, WandbLogger, ConsoleLogger
import torch.nn.utils.prune as prune



class Builder:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = torch.device(self.config.device
                                   if (torch.cuda.is_available() and
                                       self.config.device == "cuda")
                                   else "cpu")

    @staticmethod
    def fetch_remote_config(run_id: str, entity: str | None = None, project: str | None = None) -> DictConfig:
        """
        Fetches the configuration of a specific run from WandB.
        """
        if wandb is None:
            raise ImportError("WandB is not installed/active.")
        
        # Resolve entity/project if not provided (requires active run or defaults)
        # If no active run, user must likely provide them or rely on WandB defaults if authenticated
        # Resolve entity/project if not provided (requires active run or defaults)
        # If no active run, user must likely provide them or rely on WandB defaults if authenticated
        if entity and project:
            path = f"{entity}/{project}/{run_id}"
        elif project:
             path = f"{project}/{run_id}"
        else:
            path = run_id
        
        try:
            api = wandb.Api()
            run = api.run(path)
            # remote config is in run.config
            # We need to convert it back to OmegaConf
            # Note: run.config is a dict where values might be {'value': X, 'desc': ...} or just X depending on API version?
            # Standard api.run().config returns a simple dict of values.
            return OmegaConf.create(run.config)
        except Exception as e:
            raise ValueError(f"Failed to fetch config for run {run_id}: {e}")


    def build_agent(self, num_actions: int, input_channels: int, pruner: PrunerProtocol | None) -> AgentProtocol:
        network = self.build_network(num_actions, input_channels)
        optimizer = optim.Adam(network.parameters(),
            lr=self.config.algorithm.optimizer.lr,
            eps=self.config.algorithm.optimizer.eps)

        # Transfer Learning: Load weights if configured
        if self.config.get("transfer", None) and self.config.transfer.source_run_id:
            self._apply_transfer(network, self.config.transfer)

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
                encoder = NatureCNN(input_channels=input_channels,
                                    hidden_dim=self.config.algorithm.network.hidden_dim)
            case 'minatar_cnn':
                encoder = MinAtarCNN(input_channels=input_channels,
                                     hidden_dim=self.config.algorithm.network.hidden_dim)
            case _:
                raise ValueError(f"Unknown encoder: {self.config.algorithm.network.encoder}")

        match self.config.algorithm.network.head:
            case 'linear':
                head = LinearHead(encoder.output_dim,
                                  num_actions)
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

                match self.config.pruning.scheduler:
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
                # Lottery Ticket Hypothesis Pruner
                lottery_config = LotteryConfig(
                    final_sparsity=self.config.pruning.final_sparsity,
                    first_iteration_steps=self.config.train.total_timesteps,
                    rewind_to_step=self.config.pruning.get("rewind_to_step", 0),
                    pruning_rate=self.config.pruning.get("pruning_rate", 0.1),
                    iqm_window_size=self.config.pruning.get("iqm_window_size", 100)
                )
                pruner = LotteryPruner(lottery_config)
            case _:
                raise ValueError(f"Unknown pruner: {self.config.pruning.method}")

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
            log_interval=self.config.train.log_interval,
            batch_size=self.config.algorithm.batch_size,
            seed=self.config.seed,
            delegate_stopping=self.config.train.get("delegate_stopping", False)
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
            run_id=self.config.wandb.get("id"), # Handled by checkpointer/wandb logic if None
            entity=self.config.wandb.get("entity"),
            project=self.config.wandb.get("project")
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

    def _apply_transfer(self, network: QNetwork, transfer_config: DictConfig) -> None:
        """
        Loads parameters from a source run into the current network.
        Handles mask restoration for pruned models.
        """
        run_id = transfer_config.source_run_id
        alias = transfer_config.source_artifact_alias
        
        print(f"🚀 Transfer: Initializing from run {run_id} ({alias})")
        
        checkpoint_path = Checkpointer.download_checkpoint(
            run_id=run_id,
            artifact_alias=alias
        )
        
        if not checkpoint_path:
            print(f"⚠️ Transfer: Failed to download checkpoint. skipping transfer.")
            return

        # Load state dict (safely)
        try:
             # Use safe_globals if available, or just load
            from rlp.core.trainer import TrainingConfig
            if hasattr(torch.serialization, 'safe_globals'):
                with torch.serialization.safe_globals([TrainingConfig]):
                    state_dict = torch.load(checkpoint_path, map_location=self.device)
            else:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            print(f"⚠️ Transfer: Error loading checkpoint: {e}")
            return
            
        # Extract network state
        if "network" not in state_dict:
            print("⚠️ Transfer: Checkpoint does not contain 'network' key.")
            return
            
        source_net_state = state_dict["network"]
        
        # Load Encoder
        if transfer_config.load_encoder:
            print(f"📥 Transfer: Loading Encoder weights...")
            self._load_module_with_masks(network.encoder, source_net_state, prefix="encoder.")
            
        # Load Head
        if transfer_config.load_head:
            print(f"📥 Transfer: Loading Head weights...")
            self._load_module_with_masks(network.head, source_net_state, prefix="head.")

    def _load_module_with_masks(self, module: torch.nn.Module, source_state_dict: dict, prefix: str) -> None:
        """
        Loads state_dict into module, applying pruning masks if detected in source.
        """
        # 1. Detect if source has masks for this module
        # Keys in source_state_dict are like "encoder.0.weight_orig", "encoder.0.weight_mask"
        
        # Iterate over submodules of the target module to see if they match source keys
        for name, submodule in module.named_modules():
            # Construct full key path in source
            # e.g. if prefix is "encoder.", and submodule name is "0", full is "encoder.0"
            if name:
                full_name = f"{prefix}{name}"
            else:
                full_name = prefix.rstrip(".") # Top level?
            
            # Check for mask presence
            weight_mask_key = f"{full_name}.weight_mask"
            if weight_mask_key in source_state_dict:
                print(f"   ✂️ Restoring mask for {full_name}...")
                
                # We simply apply identity pruning to create the _mask and _orig buffers
                # This makes the module 'pruned' and compatible with the incoming state dict
                prune.identity(submodule, 'weight')
                
        # 2. Load the weights (strict=False because we might be loading only a subset)
        # We need to filter source_state_dict to only include keys for this module
        # to avoid "unexpected key" errors if we used strict=True, but strict=False is easier.
        # However, to be safe, we can just load.
        
        # Filter keys that start with prefix
        module_state = {k.replace(prefix, ""): v 
                        for k, v in source_state_dict.items() 
                        if k.startswith(prefix)}
        
        missing, unexpected = module.load_state_dict(module_state, strict=False)
        
        if missing:
            # Filter out expected missing keys (like if we didn't load head)
            # But here we are loading INTO 'module' (e.g. encoder).
            # So missing keys here are actual missing params in encoder.
            print(f"   ⚠️ Missing keys in {prefix[:-1]}: {missing}")
        # unexpected keys might be from other parts of network if we didn't filter perfect, 
        # but we filtered by prefix so should be fine.