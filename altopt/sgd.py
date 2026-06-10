"""
Stochastic Gradient Descent phase optimizer.

Provides fine-grained, per-sample (or micro-batch) gradient-based updates
that capture cross-block interactions missed by the ALS block-wise solver.

Key differences from standard optimizers:
  - Operates on the SAME parameters as ALS (full-rank by default)
  - Tracks gradient norm for convergence monitoring
  - Supports micro-batch accumulation for memory-constrained settings
  - Designed to alternate with ALS and Perturbation phases
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SGDPhaseOptimizer:
    """
    SGD optimizer for the SGD phase of alternating optimization.

    Thin wrapper around torch.optim.SGD with additional tracking for
    gradient norms and seamless alternation with ALS blocks.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        micro_batch_size: Optional[int] = None,
    ):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.micro_batch_size = micro_batch_size

        self.last_grad_norm: float = 0.0

        # Create optimizer over trainable parameters
        self._optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    def set_lr(self, lr: float) -> None:
        """Update learning rate (e.g., between cycles)."""
        self.lr = lr
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Perform one SGD step.

        Args:
            batch: dict with model inputs (passed directly to model.forward)

        Returns:
            loss: float scalar loss
        """
        self._optimizer.zero_grad()

        # Forward pass
        device = next(self.model.parameters()).device
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = self.model(**batch_on_device)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # Backward
        loss.backward()

        # Track gradient norm
        self.last_grad_norm = self._compute_grad_norm()

        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            max_norm=1.0,
        )

        self._optimizer.step()

        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def _compute_grad_norm(self) -> float:
        """Compute total gradient L2 norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        return total_norm ** 0.5

    def state_dict(self) -> dict:
        """Get optimizer state for checkpointing."""
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from checkpoint."""
        self._optimizer.load_state_dict(state_dict)
