import hydra
from omegaconf import DictConfig, OmegaConf

from rlp.core.builder import Builder
from rlp.core.trainer import Trainer


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.train.get('resume', False):
        cfg = load_resume_config(cfg)
        
    print(OmegaConf.to_yaml(cfg))
    
    trainer, resume_state = build_trainer(cfg)
    print('Initialized trainer.')

    if cfg.pruning.method == 'lth':
        lth(cfg, trainer, resume_state)
    else:
        print('Starting Standard Training...')
        trainer.train()

def lth(cfg: DictConfig, trainer: Trainer, resume_state: dict) -> None:
    from rlp.pruning.lottery import Lottery, LotteryConfig

    print("Initializing Lottery Ticket Hypothesis pipeline...")

    lottery_cfg = LotteryConfig(
        final_sparsity=cfg.pruning.lottery.final_sparsity,
        num_rounds=cfg.pruning.lottery.num_rounds,
        rewind_to_step=cfg.pruning.lottery.get('rewind_to_step', 0)
    )

    lottery = Lottery(trainer, lottery_cfg, resume_state=resume_state)
    lottery.run()

def load_resume_config(cfg: DictConfig) -> DictConfig:
    run_id = cfg.wandb.get("id")
    if not run_id:
        raise ValueError("Configuration specifies 'resume=True' but 'wandb.id' is missing.")

    print(f"Resuming Run {run_id}: Fetching remote configuration...")
    
    remote_cfg = Builder.fetch_remote_config(
        run_id=run_id,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project
    )
    
    final_cfg = remote_cfg
    
    final_cfg.train.resume = True
    final_cfg.wandb.id = run_id
    
    return final_cfg
    

def build_trainer(config: DictConfig) -> tuple[Trainer, dict | None]:
    builder = Builder(config)
    
    checkpointer = builder.build_checkpointer()

    start_step = 0
    state = None
    
    if config.train.resume:
        state = checkpointer.load(builder.device)
        if state:
            start_step = state['step'] + 1
            print(f"State loaded. Resuming from step {start_step}")
        else:
             print("Resume requested but no checkpoint found. Starting from scratch.")

    logger = builder.build_logger()

    ctx = builder.build_training_context(logger)

    if state and 'agent' in state:
        ctx.agent.load_state_dict(state['agent'])
    
    training_config = builder.build_training_config()
    
    return Trainer(ctx, training_config, checkpointer, start_step=start_step), state

if __name__ == "__main__":
    main()
