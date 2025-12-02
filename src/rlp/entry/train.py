import hydra
from omegaconf import DictConfig, OmegaConf
import os

from rlp.core.logger import WandbLogger, ConsoleLogger
from rlp.core.trainer import Trainer

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve config
    print(OmegaConf.to_yaml(cfg))
    
    # Setup Logger
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            job_type=cfg.wandb.job_type,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            enabled=True
        )
    else:
        logger = ConsoleLogger()
        
    # Setup Trainer
    trainer = Trainer(cfg, logger)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
