"""
Alternating Optimization Framework (AltOpt).

A post-training parameter update strategy combining:
  - ALS (Alternating Least Squares): block-wise exact solving
  - SGD (Stochastic Gradient Descent): fine-grained convergence
  - Stochastic Perturbation: local minima escape

Compared against LoRA through a 2x2 factorial protocol that
disentangles optimizer effect from parameter form effect.

Infrastructure:
  - AltOptTrainer: unified training orchestrator with hooks
  - ProfilingHub: precise FLOPs (fvcore) + memory (CUDA stats) tracking
  - PeftBridge: PEFT LoRA adapter compatibility
  - CheckpointManager: save/load/resume with hash verification
  - Evaluator: stateless unified evaluation protocol
"""

from .framework import AltOptFramework, Phase, PhaseConfig, PhaseSchedule, AltOptState
from .als import ALSBlockSolver
from .sgd import SGDPhaseOptimizer
from .perturbation import PerturbationScheduler
from .lora import LoRABaseline, LoRAConfig, LoRALayer
from .trainer import AltOptTrainer, TrainerConfig, TrainerState, run_protocol
from .profiling.flops import FlopsProfiler
from .profiling.memory import MemoryTracker
from .checkpoint import CheckpointManager
from .evaluation import Evaluator

try:
    from .peft_bridge import PeftBridge, AdapterInfo
    _has_peft = True
except ImportError:
    _has_peft = False
    PeftBridge = None  # type: ignore
    AdapterInfo = None  # type: ignore

try:
    from .model_utils import (
        ModelLoadConfig, load_model_and_tokenizer,
        enable_gradient_checkpointing, get_model_size_gb,
        estimate_training_memory_gb,
    )
    _has_model_utils = True
except ImportError:
    _has_model_utils = False
    ModelLoadConfig = None  # type: ignore
    load_model_and_tokenizer = None  # type: ignore

try:
    from .deepspeed_engine import (
        DeepSpeedConfig, DeepSpeedEngine, estimate_zero_memory,
    )
    _has_deepspeed = True
except ImportError:
    _has_deepspeed = False
    DeepSpeedConfig = None  # type: ignore
    DeepSpeedEngine = None  # type: ignore

__all__ = [
    # Core framework
    "AltOptFramework", "Phase", "PhaseConfig", "PhaseSchedule", "AltOptState",
    "ALSBlockSolver", "SGDPhaseOptimizer", "PerturbationScheduler",
    # LoRA
    "LoRABaseline", "LoRAConfig", "LoRALayer",
    # Trainer
    "AltOptTrainer", "TrainerConfig", "TrainerState", "run_protocol",
    # Profiling
    "FlopsProfiler", "MemoryTracker",
    # Infrastructure
    "CheckpointManager", "Evaluator",
    # PEFT bridge (optional)
    "PeftBridge", "AdapterInfo",
    # Model utilities (optional)
    "ModelLoadConfig", "load_model_and_tokenizer",
    # DeepSpeed engine (optional)
    "DeepSpeedConfig", "DeepSpeedEngine", "estimate_zero_memory",
]
