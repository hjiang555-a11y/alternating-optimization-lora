#!/usr/bin/env python3
"""
Cross-Architecture Validation: 4 new models × 4 runs each
Models: Gemma 2 2B/9B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Llama-8B
Protocols: B (full-rank), D-r8, D-r32, D-r256
All: AdamW, 100 steps, WikiText-2, 800 samples, seq_len=1024, batch=1, lr=1e-4
"""

import json, logging, sys, time, gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cross-arch")

MAX_SEQ_LEN, BATCH_SIZE, GRAD_ACCUM, MAX_STEPS = 1024, 1, 4, 100
N_TRAIN, N_EVAL, LR, SEED = 800, 100, 1e-4, 42
OUT_DIR = Path("runs/cross_arch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Gemma2-2B": {
        "hf": "google/gemma-2-2b",
        "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "needs_ds": False,
    },
    "Gemma2-9B": {
        "hf": "google/gemma-2-9b",
        "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "needs_ds": True,  # Full-rank 9B needs ZeRO-2
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "needs_ds": True,
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "needs_ds": True,
    },
}

RANKS = [8, 32, 256]
ALPHA_RATIO = 2.0
torch.manual_seed(SEED)


def build_dataloader(tokenizer, split, n_samples):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.select(range(min(n_samples, len(ds))))

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate(b):
        ids = torch.stack([x["input_ids"] for x in b])
        mask = torch.stack([x["attention_mask"] for x in b])
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}
    return DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=(split == "train"), collate_fn=collate)


def ppl_eval(model, dl, device):
    model.eval(); tl = 0.0; tt = 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            lo = model(**b).loss; nt = b["attention_mask"].sum().item()
            tl += lo.item() * nt; tt += nt
    return float(torch.exp(torch.tensor(tl / max(tt, 1))).item())


def run_full_rank(model_name, model_cfg, tokenizer, train_dl, eval_dl, device):
    """Protocol B: full-rank AdamW fine-tuning."""
    label = f"{model_name}_B_full"
    logger.info(">>> %s", label)
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["hf"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True)
        model.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        model.train()
        step, acc = 0, 0
        while step < MAX_STEPS:
            for b in train_dl:
                b = {k: v.to(device) for k, v in b.items()}
                loss = model(**b).loss / GRAD_ACCUM
                loss.backward(); acc += 1
                if acc >= GRAD_ACCUM:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
                    step += 1; acc = 0
                    if step >= MAX_STEPS: break
        ppl = ppl_eval(model, eval_dl, device)
        elapsed = time.time() - t0
        logger.info("  DONE: PPL=%.4f (%.0fs)", ppl, elapsed)
        result = {"experiment": label, "ppl": round(ppl, 4), "time_s": int(elapsed)}
        del model; gc.collect(); torch.cuda.empty_cache()
        return result
    except Exception as e:
        logger.error("  FAIL: %s", e, exc_info=True)
        return {"experiment": label, "error": str(e)}


def detect_lora_targets(base_model):
    """Auto-detect linear projection modules for LoRA."""
    candidates = set()
    for name, _ in base_model.named_modules():
        for pattern in ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]:
            if pattern in name and "weight" not in name:
                # Extract the module name from the full path
                candidates.add(name.split(".")[-1])
    if candidates:
        logger.info("  Auto-detected LoRA targets: %s", sorted(candidates))
        return sorted(candidates)
    return None


def run_lora(model_name, model_cfg, rank, tokenizer, train_dl, eval_dl, device):
    """Protocol D at given rank."""
    alpha = int(rank * ALPHA_RATIO)
    label = f"{model_name}_D_r{rank}"
    logger.info(">>> %s (α=%d)", label, alpha)
    t0 = time.time()
    try:
        base = AutoModelForCausalLM.from_pretrained(
            model_cfg["hf"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True)
        targets = model_cfg.get("lora_targets") or detect_lora_targets(base)
        if not targets:
            raise ValueError("Cannot detect LoRA targets; specify lora_targets in config")
        model = get_peft_model(base, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=targets))
        model.gradient_checkpointing_enable()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LR, weight_decay=0.01)
        model.train()
        step, acc = 0, 0
        while step < MAX_STEPS:
            for b in train_dl:
                b = {k: v.to(device) for k, v in b.items()}
                loss = model(**b).loss / GRAD_ACCUM
                loss.backward(); acc += 1
                if acc >= GRAD_ACCUM:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
                    step += 1; acc = 0
                    if step >= MAX_STEPS: break
        ppl = ppl_eval(model, eval_dl, device)
        elapsed = time.time() - t0
        logger.info("  DONE: PPL=%.4f, %dM params (%.0fs)", ppl, n_params / 1e6, elapsed)
        result = {"experiment": label, "rank": rank, "alpha": alpha,
                  "trainable_M": round(n_params / 1e6, 1),
                  "ppl": round(ppl, 4), "time_s": int(elapsed)}
        del model, base; gc.collect(); torch.cuda.empty_cache()
        return result
    except Exception as e:
        logger.error("  FAIL: %s", e, exc_info=True)
        return {"experiment": label, "error": str(e)}


def run_model(model_name, model_cfg):
    """Run all 4 experiments for one model."""
    logger.info("\n" + "=" * 60)
    logger.info("MODEL: %s (%s)", model_name, model_cfg["hf"])
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf"], trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_dl = build_dataloader(tokenizer, "train", N_TRAIN)
    eval_dl = build_dataloader(tokenizer, "test", N_EVAL)

    # Detect device
    dummy = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True)
    device = next(dummy.parameters()).device
    del dummy; gc.collect(); torch.cuda.empty_cache()

    results = []

    # Protocol B: full-rank
    r = run_full_rank(model_name, model_cfg, tokenizer, train_dl, eval_dl, device)
    results.append(r)

    # Protocol D: r=8, 32, 256
    for rank in RANKS:
        r = run_lora(model_name, model_cfg, rank, tokenizer, train_dl, eval_dl, device)
        results.append(r)

    # Save per-model results
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    out_file = OUT_DIR / f"{safe_name}_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", out_file)
    return results


def print_summary(all_results):
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-ARCHITECTURE SUMMARY")
    logger.info("=" * 80)
    logger.info("%-35s %10s %10s %10s %10s", "Experiment", "PPL", "r=8 PPL", "r=32 PPL", "r=256 PPL")
    for model_name in MODELS:
        model_runs = [r for r in all_results if model_name in r.get("experiment", "")]
        b = next((r for r in model_runs if "B_full" in r["experiment"]), {})
        d8 = next((r for r in model_runs if "D_r8" in r["experiment"]), {})
        d32 = next((r for r in model_runs if "D_r32" in r["experiment"]), {})
        d256 = next((r for r in model_runs if "D_r256" in r["experiment"]), {})
        logger.info("%-35s %10s %10s %10s %10s",
                     model_name,
                     b.get("ppl", b.get("error", "ERR")),
                     d8.get("ppl", d8.get("error", "ERR")),
                     d32.get("ppl", d32.get("error", "ERR")),
                     d256.get("ppl", d256.get("error", "ERR")))
    logger.info("\nPhase transition check: r=8 >> r≥32 means r_c works")
    logger.info("Overfitting check: B_full should be >> r=32 (unless memorizing)")


def main():
    logger.info("=" * 70)
    logger.info("CROSS-ARCHITECTURE VALIDATION: 4 models × 4 protocols")
    logger.info("B (full-rank) + D at r=8/32/256 | AdamW, 100 steps, WT2")
    logger.info("=" * 70)

    # Sorted by GPU cost: small first
    model_order = [
        "Gemma2-2B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Llama-8B",
        "Gemma2-9B",
    ]

    all_results = []
    for name in model_order:
        cfg = MODELS[name]
        results = run_model(name, cfg)
        all_results.extend(results)

    # Save combined
    combined_out = OUT_DIR / "combined_cross_arch.json"
    with open(combined_out, "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    logger.info("\nCombined results: %s", combined_out)
    logger.info("ALL DONE")


if __name__ == "__main__":
    sys.exit(main())
