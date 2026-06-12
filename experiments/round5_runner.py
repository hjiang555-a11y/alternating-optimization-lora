"""
Round 5: OPT-125m, 200 steps, 3 seeds, 4 protocols.

Goal: produce statistically meaningful 2x2 factorial results
with cross-seed error bars, at sufficient step count to observe
AltOpt's convergence trajectory past the ALS reconstruction loss phase.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.trainer import AltOptTrainer, TrainerConfig
from altopt.framework import Phase, PhaseConfig, PhaseSchedule
from altopt.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("round5")


MODEL_NAME = "facebook/opt-125m"
DATASET_NAME = "wikitext-2-raw-v1"
MAX_LEN = 128
BATCH_SIZE = 2
N_SAMPLES_TRAIN = 400
N_STEPS = 200
SEEDS = [42, 123, 456]
OUTPUT_DIR = Path("runs/round5_opt125m_200steps")


def make_dataloader(tokenizer, split, max_len, batch_size, n_samples=None):
    dataset = load_dataset("wikitext", DATASET_NAME, split=split)
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=max_len,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attn = torch.stack([item["attention_mask"] for item in batch])
        return {"input_ids": input_ids, "attention_mask": attn, "labels": input_ids.clone()}

    return DataLoader(tokenized, batch_size=batch_size, shuffle=(split == "train"),
                      collate_fn=collate_fn)


def run_one(protocol, opt_type, param_form, seed, tokenizer, train_dl, eval_dl, phase_schedule=None):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    t0 = time.time()

    lora_target = ["q_proj", "v_proj", "k_proj", "out_proj"]

    cfg = TrainerConfig(
        protocol=protocol,
        optimizer_type=opt_type,
        parameter_form=param_form,
        max_steps=N_STEPS,
        lr=1e-4,
        lora_r=8,
        lora_alpha=16.0,
        lora_target_modules=lora_target if param_form == "lora" else None,
        run_dir=f"/tmp/round5_{protocol}_s{seed}",
        seed=seed,
        eval_every=50,
        save_every=10000,
    )
    if phase_schedule:
        cfg.phase_schedule = phase_schedule

    trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
    state = trainer.train(train_dl)
    eval_result = Evaluator(["perplexity", "loss"], eval_dl).evaluate(model)

    train_losses = [
        l for i, l in enumerate(state.loss_history)
        if i >= len(getattr(state, 'loss_types', [])) or
        getattr(state, 'loss_types', [])[i] != 'noise_energy'
    ]

    return {
        "protocol": protocol,
        "seed": seed,
        "final_train_loss": train_losses[-1] if train_losses else float("inf"),
        "final_eval_loss": eval_result.get("loss", float("inf")),
        "final_perplexity": eval_result.get("perplexity", float("inf")),
        "total_flops": state.cumulative_flops,
        "peak_memory_mb": state.peak_memory_mb,
        "elapsed_seconds": state.elapsed_seconds,
        "wall_time": time.time() - t0,
        "loss_history": state.loss_history,
        "eval_history": state.eval_history,
        "n_steps": state.step,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_dl = make_dataloader(tokenizer, "train", MAX_LEN, BATCH_SIZE, N_SAMPLES_TRAIN)
    eval_dl = make_dataloader(tokenizer, "test", MAX_LEN, BATCH_SIZE, n_samples=100)

    altopt_schedule = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
            PhaseConfig(phase=Phase.SGD, steps=50, lr=1e-4),
            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
        ],
        cycles=4,
    )

    lora_altopt_schedule = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.SGD, steps=50, lr=1e-4),
            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=5e-4),
        ],
        cycles=4,
    )

    protocols = [
        ("A", "altopt", "full_rank", altopt_schedule),
        ("B", "adamw", "full_rank", None),
        ("C", "altopt", "lora", lora_altopt_schedule),
        ("D", "adamw", "lora", None),
    ]

    all_results = {}
    total_start = time.time()

    for protocol, opt_type, param_form, schedule in protocols:
        seed_results = {}
        for seed in SEEDS:
            label = f"{protocol}_s{seed}"
            logger.info("=" * 60)
            logger.info("Running Protocol %s (%s, %s) seed=%d",
                        protocol, opt_type, param_form, seed)
            logger.info("=" * 60)

            try:
                r = run_one(protocol, opt_type, param_form, seed,
                           tokenizer, train_dl, eval_dl, schedule)
                seed_results[str(seed)] = r
                logger.info("DONE %s s%d: ppl=%.2f flops=%.2e time=%.0fs",
                            protocol, seed, r["final_perplexity"],
                            r["total_flops"], r["wall_time"])
            except Exception as e:
                logger.error("FAILED %s s%d: %s", protocol, seed, e)
                seed_results[str(seed)] = {"error": str(e)}

        # Compute cross-seed stats
        ppls = [r["final_perplexity"] for r in seed_results.values()
                if "error" not in r and r["final_perplexity"] != float("inf")]
        losses = [r["final_eval_loss"] for r in seed_results.values()
                  if "error" not in r and r.get("final_eval_loss", float("inf")) != float("inf")]

        protocol_summary = {
            "protocol": protocol,
            "optimizer": opt_type,
            "parameter_form": param_form,
            "seeds": seed_results,
            "mean_ppl": float(np.mean(ppls)) if ppls else float("inf"),
            "std_ppl": float(np.std(ppls)) if len(ppls) > 1 else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else float("inf"),
            "std_loss": float(np.std(losses)) if len(losses) > 1 else 0.0,
        }
        all_results[protocol] = protocol_summary

        logger.info("Protocol %s SUMMARY: ppl=%.2f±%.2f (mean±std across %d seeds)",
                    protocol, protocol_summary["mean_ppl"],
                    protocol_summary["std_ppl"], len(ppls))

    total_time = time.time() - total_start

    # Final summary
    summary = {
        "experiment": "round5",
        "model": MODEL_NAME,
        "n_steps": N_STEPS,
        "n_seeds": len(SEEDS),
        "total_wall_time_s": total_time,
        "protocols": all_results,
        "comparison_matrix": {
            "A_vs_B": {
                "delta_ppl": all_results["A"]["mean_ppl"] - all_results["B"]["mean_ppl"],
                "interpretation": "Optimizer effect (full-rank)",
            },
            "C_vs_D": {
                "delta_ppl": all_results["C"]["mean_ppl"] - all_results["D"]["mean_ppl"],
                "interpretation": "Optimizer effect (LoRA)",
            },
            "A_vs_C": {
                "delta_ppl": all_results["A"]["mean_ppl"] - all_results["C"]["mean_ppl"],
                "interpretation": "Parameter form effect (AltOpt)",
            },
            "B_vs_D": {
                "delta_ppl": all_results["B"]["mean_ppl"] - all_results["D"]["mean_ppl"],
                "interpretation": "Parameter form effect (AdamW)",
            },
            "interaction_AB_CD": {
                "value": (all_results["A"]["mean_ppl"] - all_results["B"]["mean_ppl"]) -
                         (all_results["C"]["mean_ppl"] - all_results["D"]["mean_ppl"]),
                "interpretation": "Interaction: does optimizer effect depend on parameter form?",
            },
        },
    }

    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("ROUND 5 COMPLETE (%.0fs total)", total_time)
    logger.info("=" * 60)
    for protocol in ["A", "B", "C", "D"]:
        s = all_results[protocol]
        logger.info("Protocol %s (%s/%s): ppl=%.2f±%.2f",
                    protocol, s["optimizer"], s["parameter_form"],
                    s["mean_ppl"], s["std_ppl"])

    mat = summary["comparison_matrix"]
    logger.info("Optimizer effect (full-rank): Δppl_A-B = %.2f", mat["A_vs_B"]["delta_ppl"])
    logger.info("Optimizer effect (LoRA):       Δppl_C-D = %.2f", mat["C_vs_D"]["delta_ppl"])
    logger.info("Param form effect (AltOpt):    Δppl_A-C = %.2f", mat["A_vs_C"]["delta_ppl"])
    logger.info("Param form effect (AdamW):     Δppl_B-D = %.2f", mat["B_vs_D"]["delta_ppl"])
    logger.info("Interaction:                   (A-B)-(C-D) = %.2f",
                mat["interaction_AB_CD"]["value"])
    logger.info("Results saved to %s", output_path)

    return summary


if __name__ == "__main__":
    main()
