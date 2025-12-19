from __future__ import annotations

import numpy as np


class ActionTracker:
    """
    Tracks and aggregates action usage statistics for diagnostics.
    
    Maintains cumulative and recent counts of discrete actions taken by the agent.
    Useful for detecting exploration failures (e.g. rarely choosing FIRE) or
    policy collapse (e.g. always choosing NOOP).
    """
    def __init__(self, action_meanings: list[str]) -> None:
        self.meanings = action_meanings
        self.num_actions = len(action_meanings)
        
        # Cumulative counts since start
        self._total_counts = np.zeros(self.num_actions, dtype=np.int64)
        
        # Recent counts since last log
        self._recent_counts = np.zeros(self.num_actions, dtype=np.int64)
        
        self.total_steps = 0

    def update(self, actions: np.ndarray) -> None:
        """
        Update counts with a batch of actions.
        
        Args:
            actions: Array of action indices (e.g. from VectorEnv).
        """
        # Efficient bin counting for the batch
        # minlength ensures we cover all actions even if some are missing in this batch
        counts = np.bincount(actions, minlength=self.num_actions)
        
        self._total_counts += counts
        self._recent_counts += counts
        self.total_steps += len(actions)

    def get_metrics(self) -> dict[str, float]:
        """
        Return dictionary of metrics for logging.
        Resets recent counts.
        """
        metrics = {}
        
        # Total counts (cumulative)
        for i, meaning in enumerate(self.meanings):
            metrics[f"actions/total/{meaning}"] = float(self._total_counts[i])
            
        # Recent distribution (normalized frequency) or raw counts?
        # User asked for "Action counting", raw counts are safest. 
        # But frequency is easier to read across different logging intervals.
        # Let's provide raw counts for "recent" as well to be explicit.
        
        recent_total = self._recent_counts.sum()
        if recent_total > 0:
            for i, meaning in enumerate(self.meanings):
                # Log raw count
                metrics[f"actions/recent_count/{meaning}"] = float(self._recent_counts[i])
                # Log frequency (percentage) -> often more useful to spot "100% NOOP"
                metrics[f"actions/recent_freq/{meaning}"] = float(self._recent_counts[i] / recent_total)
        
        # Reset recent counts accumulator
        self._recent_counts.fill(0)
        
        return metrics
