from __future__ import annotations

import numpy as np


class SparsityScheduler:
    """Base class for sparsity schedulers."""
    def get_sparsity(self, step: int, total_steps: int) -> float:
        raise NotImplementedError


class ConstantScheduler(SparsityScheduler):
    def __init__(self, sparsity: float) -> None:
        self.sparsity = sparsity

    def get_sparsity(self, step: int, total_steps: int) -> float:
        return self.sparsity


class LinearScheduler(SparsityScheduler):
    def __init__(self, initial_sparsity: float, final_sparsity: float, start_step: int, end_step: int) -> None:
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_step = start_step
        self.end_step = end_step

    def get_sparsity(self, step: int, total_steps: int) -> float:
        if step < self.start_step:
            return self.initial_sparsity
        if step > self.end_step:
            return self.final_sparsity
        
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)


class CubicScheduler(SparsityScheduler):
    """Zhu & Gupta (2017) cubic schedule."""
    def __init__(self, initial_sparsity: float, final_sparsity: float, start_step: int, end_step: int) -> None:
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_step = start_step
        self.end_step = end_step

    def get_sparsity(self, step: int, total_steps: int) -> float:
        if step < self.start_step:
            return self.initial_sparsity
        if step > self.end_step:
            return self.final_sparsity
        
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1 - progress)**3
