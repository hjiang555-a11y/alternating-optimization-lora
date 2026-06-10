"""
Model loading and preparation utilities for 7B+ language models.

Handles the practical concerns of loading large models for post-training
experiments: dtype casting (bf16/fp16), device mapping across multiple GPUs,
gradient checkpointing (activation recomputation) for memory efficiency,
and proper tokenizer setup for causal LM tasks.

Supports Llama-2, Mistral, Qwen2, and other LLaMA-family architectures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelLoadConfig:
    """Configuration for loading a pretrained LLM for post-training experiments."""

    model_name_or_path: str
    """HuggingFace model ID or local path, e.g. 'meta-llama/Llama-2-7b-hf'."""

    dtype: str = "bf16"
    """
    Model dtype. Choices:
      - 'bf16': bfloat16 — recommended for A100/H100/RTX 5090, 2× throughput vs fp32
      - 'fp16': float16 — legacy GPUs without bf16 support
      - 'fp32': float32 — for debugging, uses 2× memory
      - 'int8': 8-bit quantization via bitsandbytes — for constrained memory
      - 'int4': 4-bit quantization via bitsandbytes — extreme memory savings
    """

    device_map: str = "auto"
    """
    Device placement strategy. Choices:
      - 'auto': HuggingFace accelerate auto device mapping across visible GPUs
      - 'sequential': manual sequential sharding (safer for unusual architectures)
      - 'balanced': balanced sharding across GPUs
      - 'balanced_low_0': balanced with GPU 0 having less memory
      - 'cpu': offload to CPU (extremely slow, for debugging only)
      - None: place on single GPU (use first available)
    """

    use_flash_attention: bool = True
    """Enable FlashAttention-2 for 2-3× faster attention if available."""

    gradient_checkpointing: bool = True
    """
    Trade compute for memory: recompute activations during backward instead of
    storing them. Reduces peak memory by ~40-60% at the cost of ~20% slower
    training. Critical for 7B+ models on consumer GPUs.
    """

    trust_remote_code: bool = False
    """Allow execution of model code from HuggingFace (required for some models)."""

    token: Optional[str] = None
    """HuggingFace API token for gated models (e.g., Llama-2 requires approval)."""

    # Quantization (bitsandbytes)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    """NormalFloat4 quantization — better than standard int4 for LLM weights."""

    def __post_init__(self):
        """Resolve dtype aliases to PyTorch dtypes."""
        self._torch_dtype = self._resolve_dtype(self.dtype)

    @staticmethod
    def _resolve_dtype(dtype_str: str) -> Optional[torch.dtype]:
        mapping = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "int8": None,   # Handled by bitsandbytes
            "int4": None,   # Handled by bitsandbytes
        }
        return mapping.get(dtype_str, torch.bfloat16)


def load_model_and_tokenizer(
    config: ModelLoadConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a pretrained LLM with the specified dtype, device mapping, and
    memory optimizations.

    Memory footprint estimates for Llama-2-7B:
      - fp32: ~28 GB
      - fp16: ~14 GB
      - bf16: ~14 GB (recommended)
      - int8:  ~7 GB
      - int4:  ~4 GB

    Args:
        config: ModelLoadConfig with model name, dtype, device settings.

    Returns:
        (model, tokenizer) tuple. The model is on the configured devices
        and ready for gradient checkpointing if enabled.

    Raises:
        ValueError: If the model cannot be loaded with the requested dtype.
    """
    logger.info("Loading model: %s (dtype=%s, device_map=%s)",
                 config.model_name_or_path, config.dtype, config.device_map)

    # ── Load tokenizer ──
    # Must load BEFORE model to avoid tokenizer/model config mismatch
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        token=config.token,
    )

    # Ensure pad token exists (many LLaMA-family tokenizers lack it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (model had no pad token)")

    # ── Load model ──
    # Build kwargs based on dtype choice
    model_kwargs: dict = {
        "trust_remote_code": config.trust_remote_code,
        "token": config.token,
    }

    if config.dtype in ("int8", "int4"):
        model_kwargs["load_in_8bit"] = (config.dtype == "int8")
        model_kwargs["load_in_4bit"] = (config.dtype == "int4")
        if config.dtype == "int4":
            model_kwargs["bnb_4bit_compute_dtype"] = getattr(
                torch, config.bnb_4bit_compute_dtype, torch.bfloat16
            )
            model_kwargs["bnb_4bit_quant_type"] = config.bnb_4bit_quant_type
        # Quantized models don't use device_map in the usual way
        # They use bitsandbytes' own memory management
    else:
        model_kwargs["torch_dtype"] = config._torch_dtype
        if config.device_map is not None:
            model_kwargs["device_map"] = config.device_map

    # Flash Attention 2
    if config.use_flash_attention:
        try:
            _ = torch.nn.functional.scaled_dot_product_attention
            # PyTorch 2.0+ has native flash attention
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("FlashAttention-2 enabled")
        except Exception:
            logger.info("FlashAttention-2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )

    # ── Gradient checkpointing ──
    if config.gradient_checkpointing and not config.dtype in ("int8", "int4"):
        try:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning("Failed to enable gradient checkpointing: %s", e)

    # ── Log memory summary ──
    _log_model_info(model, config.model_name_or_path)

    return model, tokenizer


def _log_model_info(model: PreTrainedModel, name: str) -> None:
    """Log parameter count, dtype, and device summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Detect dtype
    dtypes = set()
    devices = set()
    for p in model.parameters():
        dtypes.add(str(p.dtype).split(".")[-1])
        devices.add(str(p.device))
    dtype_str = ", ".join(sorted(dtypes))
    device_str = ", ".join(sorted(devices))

    logger.info(
        "Model '%s': %s total params (%s trainable), dtype=[%s], devices=[%s]",
        name,
        _format_params(total_params),
        _format_params(trainable_params),
        dtype_str,
        device_str,
    )


def _format_params(n: int) -> str:
    """Format parameter count in human-readable form."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """
    Enable activation checkpointing on a model.
    Safe wrapper that handles models that don't support it.

    Activation checkpointing (aka gradient checkpointing) trades compute
    for memory: instead of storing intermediate activations for the backward
    pass, it recomputes them on-the-fly. This reduces peak memory by ~40-60%
    but increases computation by ~20%.

    For a 7B model on a 32GB GPU, this is typically required for batch_size >= 2
    with sequence length >= 2048.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: enabled")
    else:
        logger.warning("Model does not support gradient_checkpointing_enable()")


def get_model_size_gb(model: nn.Module) -> float:
    """
    Estimate model memory footprint in GB (parameters only, no optimizer states).
    Uses actual dtype of parameters, not assumptions.
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 3)


def estimate_training_memory_gb(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    optimizer_type: str = "adamw",
    use_gradient_checkpointing: bool = True,
) -> dict[str, float]:
    """
    Estimate GPU memory required for training.

    Components:
      1. Model weights: param_count × dtype_bytes
      2. Optimizer states: 2× for AdamW (m, v), 1× for SGD, 0× for ALS-only
      3. Activations: batch_size × seq_length × hidden_dim × n_layers × hidden_dim
         (rough heuristic; gradient checkpointing roughly halves this)
      4. Gradients: same size as weights

    Returns:
        dict with breakdown by component and total.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_gb = param_bytes / (1024 ** 3)

    # Optimizer state memory
    if optimizer_type == "adamw":
        opt_multiplier = 2.0  # m and v
    elif optimizer_type == "sgd":
        opt_multiplier = 0.0  # no state (unless momentum)
    else:
        opt_multiplier = 0.0

    opt_gb = param_gb * opt_multiplier

    # Gradient memory
    grad_gb = param_gb

    # Activation memory (very rough heuristic)
    # Typical: batch × seq × hidden × layers × hidden ~12 bytes each
    try:
        hidden_size = model.config.hidden_size if hasattr(model, "config") else 4096
        num_layers = model.config.num_hidden_layers if hasattr(model, "config") else 32
    except Exception:
        hidden_size = 4096
        num_layers = 32

    act_bytes_per_token = hidden_size * num_layers * 2  # ~2 bytes per element in fp16
    act_gb = (batch_size * seq_length * act_bytes_per_token) / (1024 ** 3)

    if use_gradient_checkpointing:
        act_gb *= 0.3  # 70% reduction with checkpointing

    total = param_gb + opt_gb + grad_gb + act_gb

    return {
        "model_params_gb": round(param_gb, 2),
        "optimizer_states_gb": round(opt_gb, 2),
        "gradients_gb": round(grad_gb, 2),
        "activations_gb": round(act_gb, 2),
        "total_estimated_gb": round(total, 2),
    }
