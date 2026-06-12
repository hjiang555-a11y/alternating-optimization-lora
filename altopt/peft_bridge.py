"""
Bridge between AltOpt framework and HuggingFace PEFT library.

Enables Protocol C: apply the ALS-SGD-Perturbation alternating optimizer
to PEFT-injected LoRA adapters, rather than our standalone LoRALayer.

Key responsibilities:
  1. Wrap a base model with PEFT LoRA via get_peft_model()
  2. Expose only LoRA adapter parameters (lora_A, lora_B) to AltOpt
  3. Forward pass delegation to the PEFT model
  4. Merge/unload for inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PEFT_AVAILABLE = False
try:
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    logger.info("peft not installed; PeftBridge requires `pip install peft`")


@dataclass
class AdapterInfo:
    """Metadata for a single LoRA adapter within a layer."""

    lora_A: nn.Parameter
    lora_B: nn.Parameter
    base_weight: torch.Tensor
    r: int
    scaling: float
    layer_name: str


def detect_target_modules(model: nn.Module) -> list[str]:
    """
    Auto-detect LoRA target modules based on model architecture.

    Scans the model's named modules and returns the appropriate target
    module names for the detected architecture. This prevents the
    hardcoded Llama defaults from silently failing on GPT-2/OPT models.

    Detection strategy:
      1. Check model.config.architectures or model_type
      2. Scan for known module name patterns
      3. Return appropriate list or empty list if LoRA is unsupported

    Returns:
        list of module name suffixes to target, or empty list if unsupported.
    """
    model_type = ""
    if hasattr(model, "config"):
        model_type = getattr(model.config, "model_type", "").lower()

    architecture_specific = {
        "gpt2": ["c_attn", "c_proj"],
        "gpt_neo": ["attn.attention", "mlp"],
        "opt": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "qwen2": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "falcon": ["query_key_value", "dense"],
        "bloom": ["query_key_value", "dense"],
        "phi": ["q_proj", "v_proj", "k_proj", "o_proj"],
    }

    if model_type in architecture_specific:
        detected = architecture_specific[model_type]
        logger.info("Architecture '%s' → target_modules=%s", model_type, detected)
        return detected

    for name, _ in model.named_modules():
        name_lower = name.lower()
        if "q_proj" in name_lower or "k_proj" in name_lower:
            logger.info("Detected Llama-style modules → target_modules=['q_proj','v_proj','k_proj','o_proj']")
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        if "c_attn" in name_lower:
            logger.info("Detected GPT-2-style modules → target_modules=['c_attn','c_proj']")
            return ["c_attn", "c_proj"]

    logger.warning(
        "Could not detect model architecture. Defaulting to Llama target_modules. "
        "If this fails, specify target_modules explicitly."
    )
    return ["q_proj", "v_proj", "k_proj", "o_proj"]


def model_supports_lora(model: nn.Module) -> bool:
    """
    Check whether a model's architecture supports standard PEFT LoRA.

    GPT-2 (Conv1D layers) returns False because PEFT can only inject
    LoRA adapters into nn.Linear modules.
    """
    if hasattr(model, "config"):
        model_type = getattr(model.config, "model_type", "").lower()
        if model_type == "gpt2":
            return False

    has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
    return has_linear


class PeftBridge:
    """
    Adapts AltOpt to operate on PEFT-injected LoRA adapters.

    Usage:
        bridge = PeftBridge(base_model, peft_config)
        model = bridge.peft_model  # use this for forward/backward
        params = list(bridge.trainable_parameters())  # pass to AltOpt
    """

    def __init__(
        self,
        base_model: nn.Module,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
    ):
        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required for PeftBridge. Install with: pip install peft")

        if target_modules is None:
            target_modules = detect_target_modules(base_model)

        peft_config = PeftLoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.peft_model: PeftModel = get_peft_model(base_model, peft_config)
        self._adapter_map: dict[str, AdapterInfo] = {}
        self._map_adapters()

    def _map_adapters(self):
        for name, module in self.peft_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue

            lora_a = module.lora_A.get("default", None) if isinstance(module.lora_A, dict) else module.lora_A
            lora_b = module.lora_B.get("default", None) if isinstance(module.lora_B, dict) else module.lora_B

            if lora_a is None or lora_b is None:
                continue

            scaling = 1.0
            if hasattr(module, "scaling"):
                s = module.scaling
                scaling = s.get("default", 1.0) if isinstance(s, dict) else s

            r_val = getattr(module, "r", {}).get("default", 8) if isinstance(getattr(module, "r", None), dict) else getattr(module, "r", 8)

            self._adapter_map[name] = AdapterInfo(
                lora_A=lora_a,
                lora_B=lora_b,
                base_weight=module.base_layer.weight,
                r=r_val,
                scaling=scaling,
                layer_name=name,
            )

        n_adapters = len(self._adapter_map)
        n_trainable = sum(p.numel() for p in self.trainable_parameters())
        logger.info("PeftBridge: %d adapter layers, %d trainable params", n_adapters, n_trainable)

    def trainable_parameters(self):
        for info in self._adapter_map.values():
            yield info.lora_A
            yield info.lora_B

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, **kwargs) -> dict:
        return self.peft_model(**kwargs)

    def merge_and_unload(self) -> nn.Module:
        return self.peft_model.merge_and_unload()

    def get_adapter_info(self, layer_name: str) -> Optional[AdapterInfo]:
        return self._adapter_map.get(layer_name)

    def all_adapter_info(self) -> dict[str, AdapterInfo]:
        return dict(self._adapter_map)
