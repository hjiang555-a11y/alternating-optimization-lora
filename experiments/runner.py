"""
Experiment runner for the 2×2 factorial comparison protocol.

Executes four protocols:
  A: Full-Rank AltOpt   (alt optimizer, full parameter form)
  B: Full-Rank AdamW    (adam optimizer, full parameter form)
  C: LoRA-AltOpt        (alt optimizer, low-rank parameter form)
  D: LoRA-AdamW         (adam optimizer, low-rank parameter form)

Each protocol is run under the same total FLOPs budget (not equal steps)
to ensure fair resource comparison despite different per-step costs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.framework import AltOptFramework, PhaseSchedule, Phase, PhaseConfig
from altopt.lora import LoRABaseline, LoRAConfig

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a single protocol run."""
    protocol: str                     # A, B, C, or D
    optimizer_type: str               # "altopt" or "adamw"
    parameter_form: str               # "full_rank" or "lora"
    model_name: str = "gpt2"
    dataset_name: str = "wikitext-2-raw-v1"
    total_budget_flops: float = 1e15
    max_steps: Optional[int] = None
    seed: int = 42
    eval_every: int = 100
    log_dir: str = "logs/"


@dataclass
class RunResult:
    """Results from a single protocol run."""
    protocol: str
    final_loss: float
    eval_perplexity: float
    total_flops: float
    peak_memory_mb: float
    wall_time_seconds: float
    loss_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "final_loss": self.final_loss,
            "eval_perplexity": self.eval_perplexity,
            "total_flops": self.total_flops,
            "peak_memory_mb": self.peak_memory_mb,
            "wall_time_seconds": self.wall_time_seconds,
            "loss_history": self.loss_history,
            "grad_norm_history": self.grad_norm_history,
        }


class ProtocolRunner:
    """
    Executes one protocol (A/B/C/D) under a unified resource budget.

    Handles:
      - Model loading with optional LoRA injection
      - Optimizer construction (AltOpt or AdamW)
      - FLOPs tracking and budget enforcement
      - Evaluation at regular intervals
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Protocol %s: device=%s", config.protocol, self.device)

    def run(self) -> RunResult:
        """Execute the protocol and return results."""
        torch.manual_seed(self.config.seed)

        # ── Load model and tokenizer ──
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # ── Load dataset ──
        dataset = load_dataset(self.config.dataset_name, split="train")
        tokenized = dataset.map(
            lambda examples: tokenizer(
                examples["text"], truncation=True, max_length=512, padding="max_length"
            ),
            batched=True,
        )
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(tokenized, batch_size=4, shuffle=True)

        # ── Construct optimizer based on protocol ──
        if self.config.parameter_form == "lora":
            lora_cfg = LoRAConfig(r=8, alpha=16.0)
            lora_baseline = LoRABaseline(model, lora_cfg)
            optimizer = lora_baseline  # LoRABaseline handles its own optimizer
        else:
            lora_baseline = None

        if self.config.optimizer_type == "altopt":
            schedule = PhaseSchedule(
                phases=[
                    PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
                    PhaseConfig(phase=Phase.SGD, steps=100, lr=1e-4),
                    PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
                ],
                cycles=3,
            )
            framework = AltOptFramework(model, schedule)
        else:
            framework = None

        # ── Training loop with FLOPs tracking ──
        start_time = time.time()
        loss_history: list[float] = []
        grad_norm_history: list[float] = []
        total_flops = 0.0
        step = 0

        for batch in dataloader:
            if self.config.optimizer_type == "altopt" and framework is not None:
                loss = framework.step(batch)
            elif lora_baseline is not None:
                loss = lora_baseline.step(batch)
            else:
                # Full-rank AdamW fallback
                loss = self._adamw_step(model, batch)

            loss_history.append(loss)
            step += 1

            # Crude FLOPs estimate: ~3×params per backward pass
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            flops_per_step = 3.0 * n_params * 4  # ×4 for batch
            total_flops += flops_per_step

            # Budget check
            if total_flops >= self.config.total_budget_flops:
                logger.info("FLOPs budget reached at step %d", step)
                break

            if self.config.max_steps and step >= self.config.max_steps:
                break

            # Evaluation
            if step % self.config.eval_every == 0:
                logger.info(
                    "Protocol %s step %d: loss=%.4f, flops=%.2e",
                    self.config.protocol, step, loss, total_flops,
                )

        wall_time = time.time() - start_time

        # ── Final evaluation ──
        eval_perplexity = self._evaluate(model, tokenizer)

        # ── Memory tracking ──
        peak_memory = 0.0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

        result = RunResult(
            protocol=self.config.protocol,
            final_loss=loss_history[-1] if loss_history else float("inf"),
            eval_perplexity=eval_perplexity,
            total_flops=total_flops,
            peak_memory_mb=peak_memory,
            wall_time_seconds=wall_time,
            loss_history=loss_history,
            grad_norm_history=grad_norm_history,
        )

        logger.info("Protocol %s complete: ppl=%.2f, flops=%.2e, time=%.0fs",
                     self.config.protocol, eval_perplexity, total_flops, wall_time)

        return result

    def _adamw_step(self, model, batch: dict) -> float:
        """Fallback full-rank AdamW step (not the primary path)."""
        from torch.optim import AdamW
        opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        opt.zero_grad()
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        opt.step()
        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def _evaluate(self, model, tokenizer) -> float:
        """Compute perplexity on a validation subset."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        eval_dataset = load_dataset(self.config.dataset_name, split="test")
        eval_subset = eval_dataset.select(range(min(1000, len(eval_dataset))))

        with torch.no_grad():
            for example in eval_subset:
                text = example["text"]
                if len(text.strip()) < 10:
                    continue
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = model(**enc, labels=enc["input_ids"])
                total_loss += outputs.loss.item() * enc["input_ids"].numel()
                total_tokens += enc["input_ids"].numel()

        model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity


def run_all_protocols(config_path: str) -> dict[str, RunResult]:
    """
    Execute all four protocols from a YAML config file.

    Returns a dict mapping protocol label ("A", "B", "C", "D") to RunResult.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results: dict[str, RunResult] = {}
    protocols = {
        "A": ("altopt", "full_rank"),
        "B": ("adamw", "full_rank"),
        "C": ("altopt", "lora"),
        "D": ("adamw", "lora"),
    }

    for label, (opt_type, param_form) in protocols.items():
        logger.info("=== Running Protocol %s: %s + %s ===", label, opt_type, param_form)
        run_cfg = RunConfig(
            protocol=label,
            optimizer_type=opt_type,
            parameter_form=param_form,
        )
        runner = ProtocolRunner(run_cfg)
        results[label] = runner.run()

        # Save individual result
        os.makedirs(run_cfg.log_dir, exist_ok=True)
        result_path = Path(run_cfg.log_dir) / f"protocol_{label}.json"
        with open(result_path, "w") as f:
            json.dump(results[label].to_dict(), f, indent=2)

    # Save combined results
    combined_path = Path("logs") / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)

    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/configs/base.yaml"
    results = run_all_protocols(config_path)

    print("\n=== Results Summary ===")
    for label, result in results.items():
        print(f"Protocol {label}: loss={result.final_loss:.4f}, "
              f"ppl={result.eval_perplexity:.2f}, "
              f"flops={result.total_flops:.2e}, "
              f"mem={result.peak_memory_mb:.0f}MB, "
              f"time={result.wall_time_seconds:.0f}s")
