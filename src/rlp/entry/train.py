import hydra
from omegaconf import DictConfig, OmegaConf

from rlp.core.builder import Builder
from rlp.core.trainer import Trainer


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. Resume Logic: If resume=True, we MUST have an ID.
    if cfg.train.resume:
        run_id = cfg.wandb.get("id")
        if not run_id:
            raise ValueError("Configuration specifies 'resume=True' but 'wandb.id' is missing.")

        print(f"🔄 Resuming Run {run_id}: Fetching remote configuration...")
        
        # Fetch remote config
        remote_cfg = Builder.fetch_remote_config(
            run_id=run_id,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project
        )
        
        # Merge: Remote config overwrites local, BUT we generally want to keep 
        # local environment-specific things if they are not in remote? 
        # User said: "load the configuration associated with that run instead and ignore all other passed configurations."
        # However, we must ensure 'resume' and 'id' are still correct if remote didn't have them (it likely didn't have resume=True stored).
        
        # We start with remote config
        final_cfg = remote_cfg
        
        # Force the critical resume flags back onto it
        final_cfg.train.resume = True
        final_cfg.wandb.id = run_id
        
        # Also ensure we respect the current seed if it was passed? 
        # Usually resuming implies EXACT same run, so same seed.
        
        cfg = final_cfg
        
    print(OmegaConf.to_yaml(cfg))
    
    trainer, resume_state = build_trainer(cfg)
    print('Initialized trainer.')
    
    # Check for Lottery Ticket Hypothesis
    if cfg.pruning.method == 'lth':
        from rlp.pruning.lottery import Lottery, LotteryConfig
        
        print("🎭 Initializing Lottery Ticket Hypothesis pipeline...")
        
        # Create Lottery Config
        # Assuming cfg.pruning.lottery exists with relevant fields
        lottery_cfg = LotteryConfig(
            final_sparsity=cfg.pruning.lottery.final_sparsity,
            num_rounds=cfg.pruning.lottery.num_rounds,
            rewind_to_step=cfg.pruning.lottery.get('rewind_to_step', 0)
        )
        
        lottery = Lottery(trainer, lottery_cfg, resume_state=resume_state)
        lottery.run()
        
    else:
        # Standard Training
        print('🚀 Starting Standard Training...')
        trainer.train()

def build_trainer(config: DictConfig) -> tuple[Trainer, dict | None]:
    builder = Builder(config)
    
    checkpointer = builder.build_checkpointer()

    start_step = 0
    state = None
    
    if config.train.resume:
        # Load state from checkpoint
        state = checkpointer.load(builder.device)
        if state:
            start_step = state['step'] + 1
            print(f"✅ State loaded. Resuming from step {start_step}")
        else:
             print("⚠️ Resume requested but no checkpoint found. Starting from scratch.")

    logger = builder.build_logger()
    
    # Context (Agent, Buffer, Envs)
    ctx = builder.build_training_context(logger)
    
    # Restore Agent State if available
    if state and 'agent' in state:
        ctx.agent.load_state_dict(state['agent'])
    
    training_config = builder.build_training_config()
    
    return Trainer(ctx, training_config, checkpointer, start_step=start_step), state

if __name__ == "__main__":
    main()
