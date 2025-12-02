from __future__ import annotations

import abc
from typing import Any, Dict, Optional

try:
    import wandb
except ImportError:
    wandb = None


class LoggerProtocol(abc.ABC):
    """Protocol for logging metrics and configuration."""

    @abc.abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of scalar metrics."""
        pass

    @abc.abstractmethod
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close the logger."""
        pass


class ConsoleLogger(LoggerProtocol):
    """Simple logger that prints to stdout."""

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # Only print a summary or specific keys to avoid spam
        # For now, just print everything for debugging if needed, or rely on TQDM in trainer
        pass

    def log_config(self, config: Dict[str, Any]) -> None:
        print("Configuration:")
        print(config)

    def close(self) -> None:
        pass


class WandbLogger(LoggerProtocol):
    """Logger implementation using Weights & Biases."""

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[list[str]] = None,
        job_type: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        if self.enabled:
            if wandb is None:
                raise ImportError("wandb is not installed. Please install it with `pip install wandb`.")
            
            wandb.init(
                project=project,
                entity=entity,
                group=group,
                tags=tags,
                job_type=job_type,
                name=name,
                config=config,
                reinit=True,
            )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        if self.enabled:
            wandb.config.update(config, allow_val_change=True)

    def close(self) -> None:
        if self.enabled:
            wandb.finish()
