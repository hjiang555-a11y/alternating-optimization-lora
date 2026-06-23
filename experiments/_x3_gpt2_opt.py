#!/usr/bin/env python3
"""X3 Phase 1: GPT-2 + OPT-125m rank curves to fill η nomogram gaps."""
import json, sys, time, gc, os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("x3")

torch.manual_seed(42)
ML, BS, GA, MS = 1024, 1, 4, 100
NTr, NEv, LR = 800, 100, 1e-4
RANKS = [4, 8, 32]
OUT = "runs/x3_nomogram"
os.makedirs(OUT, exist_ok=True)

DEFAULT_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]


def dl(tok, sp, n):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=sp)
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=ML, padding="max_length"),
                 batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    def coll(b):
        return {"input_ids": torch.stack([x["input_ids"] for x in b]),
                "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                "labels": torch.stack([x["input_ids"] for x in b])}
    return DataLoader(ds, batch_size=BS, shuffle=(sp == "train"), collate_fn=coll)


def ppl_eval(m, dl, dev):
    m.eval(); tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = m(**b).loss; nt = b["attention_mask"].sum().item(); tl += lo.item() * nt; tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 4)


def run_model(name, hf, targets, is_conv1d=False):
    logger.info("=" * 60)
    logger.info("X3: %s (%s)", name, hf)
    logger.info("=" * 60)

    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tr_dl = dl(tok, "train", NTr); ev_dl = dl(tok, "test", NEv)

    # Baseline
    base = AutoModelForCausalLM.from_pretrained(hf, trust_remote_code=False, local_files_only=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(dev)
    bl_ppl = ppl_eval(base, ev_dl, dev)
    logger.info("Baseline PPL: %.2f", bl_ppl)
    del base; gc.collect(); torch.cuda.empty_cache()

    results = []
    for rank in RANKS:
        alpha = int(rank * 2)
        logger.info(">>> %s r=%d (α=%d)", name, rank, alpha)
        t0 = time.time()

        if is_conv1d:
            # GPT-2 uses Conv1D — need built-in LoRA, not PEFT
            from altopt.lora import LoRABaseline, LoRAConfig
            base = AutoModelForCausalLM.from_pretrained(hf, trust_remote_code=False, local_files_only=True)
            base = base.to(torch.float32).to(dev)
            lora_cfg = LoRAConfig(r=rank, alpha=alpha, dropout=0.0,
                                  target_modules=["c_attn", "c_proj"])
            lora = LoRABaseline(base, lora_cfg, lr=LR)
            model = lora.model
            opt = lora.optimizer
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.train()
        else:
            base = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16,
                                                         device_map="auto", trust_remote_code=False,
                                                         local_files_only=True)
            dev = next(base.parameters()).device
            m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                                                  target_modules=targets))
            m.gradient_checkpointing_enable()
            n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            model = m
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()),
                                    lr=LR, weight_decay=0.01)
            model.train()

        step, acc = 0, 0
        while step < MS:
            for b in tr_dl:
                b = {k: v.to(dev) for k, v in b.items()}
                (model(**b).loss / GA).backward(); acc += 1
                if acc >= GA:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
                    step += 1; acc = 0
                    if step >= MS: break

        pp = ppl_eval(model, ev_dl, dev); elapsed = time.time() - t0
        logger.info("  PPL=%.4f (%dM params, %.0fs)", pp, n_params // 1_000_000, elapsed)
        results.append({"model": name, "rank": rank, "ppl": pp, "params_M": round(n_params / 1e6, 1),
                        "time_s": int(elapsed), "baseline_ppl": bl_ppl})
        del model, base
        if not is_conv1d: del m
        del opt; gc.collect(); torch.cuda.empty_cache()

    # Compute η estimate from results
    # r_min estimated as: highest rank whose ppl differs from r=32 by >10%
    # For models where all ranks at plateau → η ≈ r_min × d_h / L
    # We know r=4 is the lower bound test; if r4/r8 < 1.05, η < 150×L/d_h
    r4 = next(r for r in results if r["rank"] == 4)
    r8 = next(r for r in results if r["rank"] == 8)
    r32 = next(r for r in results if r["rank"] == 32)
    r4r8 = r4["ppl"] / r8["ppl"] if r8["ppl"] > 0 else 999
    r8r32 = r8["ppl"] / r32["ppl"] if r32["ppl"] > 0 else 999

    logger.info("  r4/r8=%.3f, r8/r32=%.3f", r4r8, r8r32)
    if r4r8 < 1.05:
        logger.info("  r=4 already at plateau → η ≤ 4×d_h/L")
    elif r8r32 < 1.05:
        logger.info("  r=8 at plateau → η ≤ 8×d_h/L, r=4 below threshold")

    # Save
    with open(f"{OUT}/{name.replace('/', '_')}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    logger.info("X3 Phase 1: GPT-2 + OPT-125m rank curves")
    all_results = {}
    all_results["GPT-2"] = run_model("GPT-2", "gpt2", DEFAULT_TARGETS, is_conv1d=True)
    all_results["OPT-125m"] = run_model("OPT-125m", "facebook/opt-125m", DEFAULT_TARGETS)
    with open(f"{OUT}/combined_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("DONE")


if __name__ == "__main__":
    sys.exit(main())
