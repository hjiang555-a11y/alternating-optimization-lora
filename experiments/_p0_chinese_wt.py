#!/usr/bin/env python3
"""
P0: Chinese WikiText — Test rank sufficiency law across languages.

Theory predicts:
  r_min(cn) ≈ r_min(en) × H_cn/H_en  (H = per-token entropy)
  Chinese has larger character set → higher H → larger r_min.
  If H_cn/H_en > ~1.3, r=8 should be INSUFFICIENT (unlike English).

Experiment:
  Qwen2.5-0.5B (multilingual), rank curve r=8, r=32, r=256.
  Chinese WikiText-103, 800 train / 100 eval, seq_len=1024, AdamW, 100 steps.

English baseline for comparison (from _xval.py, matching config):
  r=8: 1.62  r=32: 1.60  r=256: 1.61  → ALL at plateau.

Chinese prediction:
  r=8 noticeably worse than r=32 (below r_min).
  r=32 ≈ r=256 (plateau exists, just at higher rank).
"""

import json, logging, sys, time, gc, os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p0-zh")

# ── Config (IDENTICAL to English _xval.py) ──
ML, BS, GA, MS = 1024, 1, 4, 100
NTr, NEv, LR, SD = 800, 100, 1e-4, 42
torch.manual_seed(SD)
TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]
RANKS = [8, 32, 256]
OUT = "runs/p0_chinese_wt"


class WikiTextDataset(Dataset):
    """Stream WikiText-103 into fixed-length chunks without HF datasets caching issues."""

    def __init__(self, split, n_samples, tokenizer, max_len):
        # Use the Salesforce/wikitext dataset (fresh-cached at /tmp/wikitext_cache)
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        # Filter empty lines, concatenate remaining
        texts = [ex["text"] for ex in ds if ex["text"].strip()]
        # Merge into one long string, then chunk
        full_text = "\n\n".join(texts)
        encoding = tokenizer(full_text, truncation=False, return_tensors="pt")["input_ids"][0]
        # Create non-overlapping chunks of max_len
        self.chunks = []
        i = 0
        while i + max_len <= len(encoding) and len(self.chunks) < n_samples:
            self.chunks.append(encoding[i : i + max_len])
            i += max_len
        logger.info("  %s: %d chunks from %d tokens (target: %d)", split, len(self.chunks), len(encoding), n_samples)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}


def ppl_eval(model, dl, dev):
    model.eval()
    tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = model(**b).loss
            nt = b["attention_mask"].sum().item()
            tl += lo.item() * nt
            tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 4)


def main():
    logger.info("=" * 70)
    logger.info("P0: Chinese WikiText-103 — Rank Sufficiency Law Cross-Language Test")
    logger.info("r_min(cn) ≈ r_min(en) × H_cn/H_en — predicting r=8 INSUFFICIENT")
    logger.info("=" * 70)

    # Load model + tokenizer
    logger.info("Loading Qwen2.5-0.5B (multilingual)...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check tokenizer handles Chinese
    cn_test = "这是中文维基百科的一个测试句子"
    cn_tokens = tokenizer.encode(cn_test)
    logger.info("Chinese tokenization: '%s' → %d tokens", cn_test, len(cn_tokens))

    # Build datasets
    logger.info("Building Chinese WikiText-103 datasets...")
    tr_ds = WikiTextDataset("train", NTr, tokenizer, ML)
    ev_ds = WikiTextDataset("test", NEv, tokenizer, ML)
    tr_dl = DataLoader(tr_ds, batch_size=BS, shuffle=True,
                       collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                                             "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                                             "labels": torch.stack([x["input_ids"] for x in b])})
    ev_dl = DataLoader(ev_ds, batch_size=BS, shuffle=False,
                       collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                                             "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                                             "labels": torch.stack([x["input_ids"] for x in b])})

    results = []

    # Baseline PPL (untrained model on Chinese)
    logger.info(">>> Baseline PPL (Chinese)")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=False, local_files_only=True)
    dev = next(base.parameters()).device
    bl_ppl = ppl_eval(base, ev_dl, dev)
    logger.info("  BASELINE (Chinese): PPL=%.2f", bl_ppl)
    results.append({"run": "P0_zh_baseline", "ppl": bl_ppl})
    del base
    gc.collect()
    torch.cuda.empty_cache()

    # Rank curve
    for rank in RANKS:
        alpha = int(rank * 2)
        label = f"P0_zh_r{rank}"
        logger.info(">>> %s (α=%d)", label, alpha)

        base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=False, local_files_only=True)
        dev = next(base.parameters()).device
        model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                                                  target_modules=TARGETS))
        model.gradient_checkpointing_enable()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)

        t0 = time.time()
        model.train()
        step, acc = 0, 0
        while step < MS:
            for b in tr_dl:
                b = {k: v.to(dev) for k, v in b.items()}
                (model(**b).loss / GA).backward()
                acc += 1
                if acc >= GA:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    step += 1
                    acc = 0
                    if step >= MS:
                        break

        pp = ppl_eval(model, ev_dl, dev)
        elapsed = time.time() - t0
        logger.info("  DONE: PPL=%.4f (%dM params, %.0fs)", pp, n_params // 1_000_000, elapsed)
        results.append({"run": label, "rank": rank, "ppl": pp, "params_M": round(n_params / 1e6, 1),
                        "time_s": int(elapsed)})
        del model, base, opt
        gc.collect()
        torch.cuda.empty_cache()

    # ── Compare with English ──
    en_baseline = {"Qwen2.5-0.5B": 133.16}
    en_results = {"r8": 1.62, "r32": 1.60, "r256": 1.61}

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS: Chinese vs English Rank Curve")
    logger.info("%-12s %8s %8s %8s %8s", "Rank", "CN PPL", "EN PPL", "CN/EN", "r8/r32")
    logger.info("-" * 70)
    for rank in RANKS:
        cn = next(r["ppl"] for r in results if r.get("rank") == rank)
        en = en_results[f"r{rank}"]
        ratio = cn / en
        r8r32 = "—"
        if rank != 8:
            cn_r8 = next(r["ppl"] for r in results if r.get("rank") == 8)
            cn_r32_or_256 = next(r["ppl"] for r in results if r.get("rank") == rank)
            r8r32 = f"{cn_r8 / cn_r32_or_256:.3f}"
        logger.info("r%-8d %8.4f %8.4f %8.3f %8s", rank, cn, en, ratio, r8r32)

    logger.info("\nHYPOTHESIS TEST:")
    cn_r8 = next(r["ppl"] for r in results if r.get("rank") == 8)
    cn_r32 = next(r["ppl"] for r in results if r.get("rank") == 32)
    cn_r256 = next(r["ppl"] for r in results if r.get("rank") == 256)
    r8degrade = cn_r8 / cn_r32

    if r8degrade > 1.50:
        logger.info("  ✅ CONFIRMED: r=8 significantly worse than r=32 (ratio=%.2f×)", r8degrade)
        logger.info("  r_min(Chinese) > 8 — η_cn > η_en as predicted.")
    elif r8degrade > 1.10:
        logger.info("  ⚠  WEAK SUPPORT: r=8 marginally worse (ratio=%.2f×)", r8degrade)
        logger.info("  r_min(Chinese) may be ~8, near threshold.")
    else:
        logger.info("  ❌ FALSIFIED: r=8 matches r=32 (ratio=%.2f×)", r8degrade)
        logger.info("  r_min is language-independent for WikiText. η ∝ H NOT supported.")

    r32r256 = cn_r32 / cn_r256
    if r32r256 < 1.10:
        logger.info("  r=32 vs r=256: plateau exists at r≥32 (ratio=%.2f×)", r32r256)
    else:
        logger.info("  r=32 vs r=256: r=32 may also be below r_min (ratio=%.2f×)", r32r256)

    # Save
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved: %s/results.json", OUT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
