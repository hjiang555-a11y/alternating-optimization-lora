"""
Stochastic perturbation mechanisms for escaping local minima.

Implements scheduled parameter-space noise injection with configurable
distributions (Gaussian, uniform), decay schedules, and adaptive scaling.

The key insight: pure gradient methods (SGD, Adam) can get trapped in
narrow local minima. Periodic perturbation adds controlled noise to the
parameter space, allowing the optimizer to explore the surrounding loss
landscape and potentially find better basins.

This is a critical differentiator from LoRA+AdamW, which lacks explicit
perturbation mechanisms.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PerturbationScheduler:
    """
    Applies controlled stochastic noise to model parameters.

    Supports:
      - Gaussian noise: θ ← θ + ε, ε ~ N(0, σ²)
      - Uniform noise: θ ← θ + ε, ε ~ U(-σ, σ)
      - Cosine decay schedule for noise scale over cycles
      - Layer-wise scaling (different noise for different layer types)
    """

    def __init__(
        self,
        model: nn.Module,
        noise_type: str = "gaussian",
        initial_scale: float = 1e-3,
        decay_schedule: str = "cosine",
        min_scale: float = 1e-6,
        cycle: int = 0,
    ):
        self.model = model
        self.noise_type = noise_type
        self.initial_scale = initial_scale
        self.decay_schedule = decay_schedule
        self.min_scale = min_scale
        self.cycle = cycle

    def apply_noise(
        self,
        scale: Optional[float] = None,
        cycle: Optional[int] = None,
    ) -> float:
        """
        Apply perturbation noise to all trainable parameters.

        Args:
            scale: noise standard deviation (overrides schedule if provided)
            cycle: current cycle index (for decay schedule)

        Returns:
            avg_noise_energy: average ||Δθ||² across parameters (for logging)
        """
        if cycle is not None:
            self.cycle = cycle

        effective_scale = self._compute_scale(scale)

        total_energy = 0.0
        n_params = 0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                layer_scale = effective_scale * self._layer_multiplier(name)

                if self.noise_type == "gaussian":
                    noise = torch.randn_like(param) * layer_scale
                elif self.noise_type == "uniform":
                    noise = (torch.rand_like(param) * 2 - 1) * layer_scale
                else:
                    raise ValueError(f"Unknown noise_type: {self.noise_type}")

                param.add_(noise)
                energy = (noise ** 2).sum().item()
                total_energy += energy
                n_params += param.numel()

        avg_energy = total_energy / max(n_params, 1)
        logger.debug(
            "Perturb cycle=%d, scale=%.2e, avg_energy=%.2e",
            self.cycle, effective_scale, avg_energy
        )

        return avg_energy

    def _compute_scale(self, override_scale: Optional[float]) -> float:
        """Compute effective noise scale based on schedule."""
        if override_scale is not None:
            return override_scale

        if self.decay_schedule == "cosine":
            # Cosine decay: σ_c = σ_0 * 0.5 * (1 + cos(π * c / C_max))
            max_cycles = 10  # heuristic
            progress = min(self.cycle / max(max_cycles, 1), 1.0)
            scale = self.initial_scale * 0.5 * (1 + math.cos(math.pi * progress))
        elif self.decay_schedule == "exponential":
            scale = self.initial_scale * (0.5 ** self.cycle)
        elif self.decay_schedule == "constant":
            scale = self.initial_scale
        else:
            scale = self.initial_scale

        return max(scale, self.min_scale)

    @staticmethod
    def _layer_multiplier(name: str) -> float:
        """
        Scale noise differently by layer type.

        Embedding layers get less noise (semantic disruption),
        attention projections get moderate noise,
        FFN layers get more noise (more redundancy).
        """
        if "embed" in name.lower():
            return 0.1
        elif "attn" in name.lower() or "attention" in name.lower():
            return 0.5
        elif "ffn" in name.lower() or "mlp" in name.lower() or "fc" in name.lower():
            return 1.0
        else:
            return 0.5

    def reset(self, cycle: int = 0) -> None:
        """Reset cycle counter (e.g., for new training run)."""
        self.cycle = cycle
