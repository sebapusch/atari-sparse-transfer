import hydra
from omegaconf import DictConfig, OmegaConf
import os

from rlp.core.builder import Builder
from rlp.core.logger import WandbLogger, ConsoleLogger, LoggerProtocol
from rlp.core.trainer import Trainer


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
        
    trainer = build_trainer(cfg)
    
    # Train
    trainer.train()

def build_trainer(config: DictConfig) -> Trainer:
    builder = Builder(config)
    
    logger = builder.build_logger()
    ctx = builder.build_training_context(logger)
    training_config = builder.build_training_config()
    checkpointer = builder.build_checkpointer()

    return Trainer(ctx, training_config, checkpointer)

if __name__ == "__main__":
    main()
