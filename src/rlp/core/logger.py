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
        self._metric_buffer: list[tuple[dict[str, float], int | None]] = []

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Buffers metrics to be logged later."""
        self._metric_buffer.append((metrics, step))

    def log_config(self, config: dict[str, Any]) -> None:
        self.run.config.update(config, allow_val_change=True)
    
    def commit(self) -> None:
        """
        Flushes all buffered metrics to WandB. 
        Should be called when a checkpoint is saved.
        """
        if not self._metric_buffer:
            return
            
        for metrics, step in self._metric_buffer:
            self.run.log(metrics, step=step)
        
        self._metric_buffer.clear()

    def close(self) -> None:
        # Commit remaining metrics?
        # If we crashed before checkpoint, maybe we lose them? 
        # User requested "only occurs when a model checkpoint is saved".
        # So we should probably NOT commit implicitly on close, unless close implies a save?
        # But Trainer.train() calls save() at the end.
        self.run.finish()
