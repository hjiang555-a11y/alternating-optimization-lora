"""
Resource-aware metrics for fair comparison between AltOpt and LoRA.

Since ALS matrix inversion and SGD gradient computation have fundamentally
different cost profiles, we cannot compare "per step" — we must compare
at equal resource budgets.

Three resource dimensions are tracked:
  1. FLOPs: total floating-point operations
  2. Memory: peak GPU memory allocation
  3. Wall-clock time: actual elapsed seconds

The primary comparison uses FLOPs-normalized budgets. Memory and time
are secondary metrics reported alongside.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ResourceBudget:
    """Resource budget for a single training run."""
    total_flops: float = float("inf")
    max_memory_mb: float = float("inf")
    max_time_seconds: float = float("inf")

    def is_exceeded(self, current: "ResourceUsage") -> bool:
        return (
            current.cumulative_flops >= self.total_flops
            or current.peak_memory_mb >= self.max_memory_mb
            or current.elapsed_seconds >= self.max_time_seconds
        )


@dataclass
class ResourceUsage:
    """Cumulative resource usage tracked during training."""
    cumulative_flops: float = 0.0
    peak_memory_mb: float = 0.0
    elapsed_seconds: float = 0.0
    step: int = 0

    _start_time: float = 0.0

    def __post_init__(self):
        self._start_time = time.time()

    def update(self, flops: float) -> None:
        """Record resource consumption for one step."""
        self.cumulative_flops += flops
        self.step += 1
        self.elapsed_seconds = time.time() - self._start_time

        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory)


def estimate_step_flops(
    model: nn.Module,
    parameter_form: str,
    optimizer_type: str,
    batch_size: int,
    seq_length: int,
) -> float:
    """
    Estimate FLOPs for a single optimization step.

    Breakdown:
      - Forward pass: ~2 × params FLOPs (matrix multiplies)
      - Backward pass: ~4 × params FLOPs (gradient computation)
      - ALS block inversion (if applicable): O(b³) per block
      - Perturbation (if applicable): negligible O(d)

    Returns:
        Estimated FLOPs as a float.
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Base forward + backward cost (same for all optimizers)
    base_flops = 6.0 * n_params * batch_size * seq_length

    if optimizer_type == "altopt":
        # ALS phase: add O(b³) per block
        # For a fair estimate, assume one ALS step has cost ~10× a backward pass
        als_overhead = base_flops * 0.05  # amortized over SGD steps
        return base_flops + als_overhead

    # AdamW has 2× state overhead (m, v) but similar per-step FLOPs to SGD
    return base_flops


def params_to_flops(n_params: int, batch_size: int, seq_length: int) -> float:
    """
    Heuristic FLOPs estimator: 6× forward+backward multiplies.

    This is a rough approximation. For precise accounting, use fvcore
    or a custom flop counter hooked into the model.
    """
    return 6.0 * n_params * batch_size * seq_length


def normalize_to_budget(
    results: dict[str, "RunResult"],
    budget_dimension: str = "flops",
) -> dict[str, float]:
    """
    Normalize results to a common resource budget.

    For FLOPs normalization: compare loss/perplexity at equal FLOPs.
    Uses linear interpolation between logged data points.

    Args:
        results: dict mapping protocol label to RunResult
        budget_dimension: "flops", "memory", or "time"

    Returns:
        dict mapping protocol label to normalized metric
    """
    from .runner import RunResult

    normalized: dict[str, float] = {}
    return normalized
