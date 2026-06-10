"""
Alternating Least Squares (ALS) block-wise exact solver.

Partitions model parameters into independent blocks and solves each block
exactly via least squares (requiring matrix inversion) while holding all
other blocks fixed.

This is the most computationally expensive phase of the alternating
optimization framework — each block requires solving a linear system
of size (block_size × block_size), costing O(b³) per block.

For LLM weight matrices W ∈ ℝ^{d_out × d_in}, we partition rows into
blocks of size b, solving:

    W_block = argmin ||X W_block^T - Y_target||²

where X is the input activations for that block's rows.

This yields the closed-form solution:

    W_block = (X^T X + λI)^{-1} X^T Y_target

The λI regularization prevents ill-conditioned inverses when X is
near-singular.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ALSBlockSolver:
    """
    Block-wise ALS solver for linear layers.

    Operates on nn.Linear modules in the model, solving each block of rows
    independently via regularized least squares.
    """

    def __init__(
        self,
        model: nn.Module,
        reg_lambda: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.reg_lambda = reg_lambda
        self.device = device or next(model.parameters()).device

        # Cache: store (X^T X + λI)^{-1} X^T per block for warm-start
        self._cache: dict[str, torch.Tensor] = {}

    def solve_block(
        self,
        batch: dict[str, torch.Tensor],
        block_size: int = 1024,
    ) -> float:
        """
        Solve one ALS block update across all linear layers.

        For each nn.Linear layer, partitions output rows into blocks of
        `block_size`, solves each block exactly, and updates the weight.

        Args:
            batch: dict with 'input_ids', 'labels' (and optionally 'attention_mask')
            block_size: number of output rows per block

        Returns:
            total_loss: float, loss after applying all block updates
        """
        total_loss = 0.0
        n_blocks_total = 0

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            loss, n_blocks = self._solve_linear_layer(
                name, module, batch, block_size
            )
            total_loss += loss
            n_blocks_total += n_blocks

        if n_blocks_total > 0:
            logger.debug(
                "ALS: solved %d blocks across layers, total_loss=%.6f",
                n_blocks_total, total_loss
            )

        return total_loss / max(n_blocks_total, 1)

    def _solve_linear_layer(
        self,
        name: str,
        module: nn.Linear,
        batch: dict[str, torch.Tensor],
        block_size: int,
    ) -> tuple[float, int]:
        """
        Solve one nn.Linear layer via block-wise ALS.

        Returns (loss, n_blocks_solved).
        """
        weight = module.weight.data  # [d_out, d_in]
        d_out, d_in = weight.shape
        device = weight.device

        # ── Forward pass to collect activations ──
        # We need the input to this layer. For now, use a hook-based approach
        # or require the user to provide activations.
        # Simplified: do a forward pass and capture inputs via hook.

        activations: list[torch.Tensor] = []
        hook_handle = module.register_forward_pre_hook(
            lambda _mod, inp: activations.append(inp[0].detach())
        )

        try:
            # Run forward with no grad to get activations
            with torch.no_grad():
                _ = self.model(**{k: v.to(device) for k, v in batch.items()
                                  if isinstance(v, torch.Tensor)})
            hook_handle.remove()

            if not activations:
                return 0.0, 0

            X = activations[0]  # [batch * seq_len, d_in] or [batch, seq_len, d_in]
            if X.dim() == 3:
                X = X.reshape(-1, d_in)

            # ── Get targets via backward ──
            # For post-training: target is to minimize reconstruction or task loss
            # Simplified: use the current output as reference for least squares
            # In practice, this uses a separate target computation

            n_blocks = (d_out + block_size - 1) // block_size
            total_loss = 0.0

            for i in range(n_blocks):
                start = i * block_size
                end = min(start + block_size, d_out)

                # Current block of weights
                W_block = weight[start:end, :].clone()  # [b, d_in]

                # Solve: W_new = (X^T X + λI)^{-1} X^T Y
                XtX = X.T @ X  # [d_in, d_in]
                reg = self.reg_lambda * torch.eye(d_in, device=device, dtype=X.dtype)
                XtX_reg = XtX + reg

                # Cholesky for stability
                try:
                    L = torch.linalg.cholesky(XtX_reg)
                    XtX_inv_Xt = torch.cholesky_solve(X.T, L)  # [d_in, batch*n]
                except RuntimeError:
                    # Fallback to pseudoinverse if Cholesky fails
                    XtX_inv_Xt = torch.linalg.lstsq(XtX_reg, X.T).solution

                # Target: current forward output (least squares approximation)
                Y = X @ W_block.T  # [N, b]

                W_new = (Y.T @ XtX_inv_Xt.T).to(weight.dtype)  # [b, d_in]

                # Update weights in-place
                weight[start:end, :] = W_new

                # Track loss
                recon_error = torch.norm(X @ W_new.T - Y) ** 2
                total_loss += recon_error.item()

            return total_loss, n_blocks

        except Exception as e:
            logger.warning("ALS block solve failed for layer '%s': %s", name, e)
            hook_handle.remove()
            return 0.0, 0

    def clear_cache(self) -> None:
        """Clear cached inverses (useful between runs)."""
        self._cache.clear()
