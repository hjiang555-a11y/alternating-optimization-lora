"""
Alternating Optimization Framework (AltOpt).

A post-training parameter update strategy combining:
  - ALS (Alternating Least Squares): block-wise exact solving
  - SGD (Stochastic Gradient Descent): fine-grained convergence
  - Stochastic Perturbation: local minima escape

Compared against LoRA through a 2x2 factorial protocol that
disentangles optimizer effect from parameter form effect.
"""

from .framework import AltOptFramework, Phase, PhaseSchedule
from .als import ALSBlockSolver
from .sgd import SGDPhaseOptimizer
from .perturbation import PerturbationScheduler
from .lora import LoRABaseline, LoRAConfig

__all__ = [
    "AltOptFramework",
    "Phase",
    "PhaseSchedule",
    "ALSBlockSolver",
    "SGDPhaseOptimizer",
    "PerturbationScheduler",
    "LoRABaseline",
    "LoRAConfig",
]
