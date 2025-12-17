from __future__ import annotations

import abc
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None


class LoggerProtocol(abc.ABC):
    """Protocol for logging metrics and configuration."""

    @abc.abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dictionary of scalar metrics."""
        pass

    @abc.abstractmethod
    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration parameters."""
        pass

    def commit(self) -> None:
        """Commit buffered metrics (if any)."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close the logger."""
        pass


class ConsoleLogger(LoggerProtocol):
    """Simple logger that prints to stdout."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        print(f"[{step}] {metrics}")

    def log_config(self, config: dict[str, Any]) -> None:
        print("Configuration:")
        print(config)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


class WandbLogger(LoggerProtocol):
    """Logger implementation using Weights & Biases."""

    def __init__(self, run: wandb.Run) -> None:
        self.run = run
        # Direct logging mode, no buffering needed for live view.

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs metrics directly to WandB."""
        self.run.log(metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self.run.config.update(config, allow_val_change=True)
    
    def commit(self) -> None:
        """
        No-op for direct logging.
        """
        pass

    def close(self) -> None:
        self.run.finish()
