#!/usr/bin/env python3
"""P4: SmolLM2 fine-grained r_min. Test r=10,12,14 to pinpoint threshold."""
import json, sys, time, gc, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p4")

ML, BS, GA, MS = 1024, 1, 4, 100
NTr, NEv, LR, SD = 800, 100, 1e-4, 42
RANKS = [10, 12, 14]
TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]
OUT = "runs/p4_smollm2"

torch.manual_seed(SD)

def dl(tok, sp, n):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=sp)
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=ML, padding="max_length"),
                 batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=BS, shuffle=(sp == "train"),
                       collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                                             "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                                             "labels": torch.stack([x["input_ids"] for x in b])})

def ppl_eval(m, dl, dev):
    m.eval(); tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = m(**b).loss; nt = b["attention_mask"].sum().item(); tl += lo.item() * nt; tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 4)

def main():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tr_dl = dl(tok, "train", NTr); ev_dl = dl(tok, "test", NEv)
    base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=False, local_files_only=True)
    dev = next(base.parameters()).device
    del base; gc.collect(); torch.cuda.empty_cache()

    results = []
    for rank in RANKS:
        alpha = int(rank * 2)
        logger.info(">>> r=%d (α=%d)", rank, alpha)
        base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=False, local_files_only=True)
        dev = next(base.parameters()).device
        m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=TARGETS))
        m.gradient_checkpointing_enable()
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR, weight_decay=0.01)
        m.train(); step, acc = 0, 0; t0 = time.time()
        while step < MS:
            for b in tr_dl:
                b = {k: v.to(dev) for k, v in b.items()}; (m(**b).loss / GA).backward(); acc += 1
                if acc >= GA:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
                    step += 1; acc = 0
                    if step >= MS: break
        pp = ppl_eval(m, ev_dl, dev); el = time.time() - t0
        logger.info("  PPL=%.4f (%dM params, %.0fs)", pp, n_params // 1_000_000, el)
        results.append({"rank": rank, "ppl": pp, "params_M": round(n_params / 1e6, 1), "time_s": int(el)})
        del m, base, opt; gc.collect(); torch.cuda.empty_cache()

    # Compare with known data points
    ref = {6: 15.29, 8: 3.09, 16: 1.86, 32: 1.76, 256: 1.69}
    logger.info("\n=== SMOLLM2 COMPLETE RANK THRESHOLD MAP ===")
    logger.info("%-8s %10s", "Rank", "PPL")
    for rank in sorted(set(list(ref.keys()) + [r["rank"] for r in results])):
        pp = ref.get(rank) or next(r["ppl"] for r in results if r["rank"] == rank)
        marker = " ← NEW" if rank in RANKS else ""
        logger.info("r%-7d %10.4f%s", rank, pp, marker)

    logger.info("\nTRANSITION ANALYSIS:")
    # Find exact r_min: last rank where r to r+2 improves by >5%
    all_data = {r["rank"]: r["ppl"] for r in results}
    all_data.update(ref)
    for threshold in [12, 14]:
        below = all_data.get(threshold - 2, 999)
        at = all_data.get(threshold, 999)
        above = all_data.get(threshold + 2, 999)
        improvement = (below - at) / max(below, 0.01)
        next_improvement = (at - above) / max(at, 0.01) if above < 999 else 0
        logger.info("  r=%d: improvement from r=%d = %.1f%%", threshold, threshold-2, improvement*100)
        if improvement > 0.05 and next_improvement < 0.05:
            logger.info("  → r_min ≈ %d (threshold confirmed at this rank)", threshold)
        elif improvement > 0.10:
            logger.info("  → r=%d still significantly improving — r_min is higher", threshold)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/results.json", "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    sys.exit(main())
