#!/usr/bin/env python3
"""
Falsification experiments for Rank Sufficiency Law: r_min = η × L/d_h.

Three critical tests (highest information gain per GPU minute):
1. Mistral-7B r=4: predicted plateau (r_min ≈ 1.8-5.2, r=4 > r_min)
2. SmolLM2-135M r=16: predicted plateau (r_min ≈ 12, r=16 > r_min)
3. SmolLM2-135M r=6: predicted degradation (r_min ≈ 12, r=6 < r_min)

If (1) and (2) pass but (3) fails: η and L/d_h confirmed quantitatively.
If (1) passes: formula is dimensionally correct.
If (2) fails: r_min(SmolLM2) > 16 → η > 307, formula underestimates.
"""

import json, logging, sys, time, gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("falsify")

ML, BS, GA, MS = 1024, 1, 4, 100
NTr, NEv, LR, SD = 800, 100, 1e-4, 42
torch.manual_seed(SD)
OUT_DIR = Path("runs/falsify")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup(hf):
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ds = lambda s, n: DataLoader(
        (lambda d: d.select(range(min(n, len(d)))))(load_dataset('wikitext', 'wikitext-2-raw-v1', split=s))
        .map(lambda ex: tok(ex['text'], truncation=True, max_length=ML, padding='max_length'), batched=True, remove_columns=['text'])
        .with_format('torch', columns=['input_ids', 'attention_mask']),
        batch_size=BS, shuffle=(s == 'train'),
        collate_fn=lambda b: {'input_ids': torch.stack([x['input_ids'] for x in b]),
                              'attention_mask': torch.stack([x['attention_mask'] for x in b]),
                              'labels': torch.stack([x['input_ids'] for x in b]).clone()})
    return tok, ds('train', NTr), ds('test', NEv)


def ppl(m, dl, dev):
    m.eval(); tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = m(**b).loss; nt = b['attention_mask'].sum().item()
            tl += lo.item() * nt; tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 4)


def train(m, opt, dl, dev):
    m.train(); step, acc = 0, 0
    while step < MS:
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            (m(**b).loss / GA).backward(); acc += 1
            if acc >= GA:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                step += 1; acc = 0
                if step >= MS: break


def run_one(name, hf, rank, targets, prediction):
    alpha = int(rank * 2)
    logger.info(">>> %s r=%d (prediction: %s)", name, rank, prediction)
    tok, tr_dl, ev_dl = setup(hf)
    base = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=False,
                                                 local_files_only=True)
    dev = next(base.parameters()).device
    m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                                         target_modules=targets))
    m.gradient_checkpointing_enable()
    np_sum = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR, weight_decay=0.01)
    t0 = time.time()
    train(m, opt, tr_dl, dev)
    pp = ppl(m, ev_dl, dev); el = time.time() - t0

    result = {"experiment": f"{name}_r{rank}", "rank": rank, "ppl": pp,
              "params_M": round(np_sum / 1e6, 1), "time_s": int(el),
              "prediction": prediction}
    logger.info("  PPL=%.4f (%.0fs) — prediction: %s → %s",
                 pp, el, prediction,
                 "✓ CONFIRMED" if pp < 1.6 else "✗ FALSIFIED" if pp > 2.5 else "⚠ AMBIGUOUS")
    del m, base, opt; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    logger.info("=" * 70)
    logger.info("FALSIFICATION EXPERIMENTS: Rank Sufficiency Law")
    logger.info("r_min = η × L/d_h  (η ≈ 230)")
    logger.info("=" * 70)

    all_results = []

    # Test 1: Mistral-7B r=4 (highest leverage — tests dimensional correctness)
    all_results.append(run_one(
        "Mistral-7B", "mistralai/Mistral-7B-v0.3", 4,
        ["q_proj", "v_proj", "k_proj", "o_proj"],
        "r_min≈1.8-5.2, r=4 should be at plateau → PPL ≈ 1.45"))

    # Test 2: SmolLM2 r=16 (tests threshold position — should match r=32)
    all_results.append(run_one(
        "SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M", 16,
        ["q_proj", "v_proj", "k_proj", "o_proj"],
        "r_min≈12, r=16 > r_min → should match r=32 plateau (PPL≈1.76)"))

    # Test 3: SmolLM2 r=6 (tests below-threshold degradation — should be worse)
    all_results.append(run_one(
        "SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M", 6,
        ["q_proj", "v_proj", "k_proj", "o_proj"],
        "r_min≈12, r=6 < r_min → should show degradation (PPL≈2.2-2.5)"))

    # Save and summarize
    out_file = OUT_DIR / "falsify_results.json"
    with open(out_file, "w") as f: json.dump(all_results, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS: Rank Sufficiency Law Validation")
    logger.info("%-30s %5s %8s %10s %s", "Experiment", "PPL", "r_min", "Predicted", "Verdict")
    logger.info("-" * 70)
    for r in all_results:
        exp = r["experiment"]
        pp = r["ppl"]
        r_min = "—"
        if "Mistral" in exp: r_min = "1.8-5.2"
        elif "SmolLM2" in exp: r_min = "12"
        pred = "plateau" if ("r=4" in exp or "r=16" in exp) else "degrade"
        verdict = "✓" if (("r=4" in exp or "r=16" in exp) and pp < 2.0) or ("r=6" in exp and pp > 2.0) else "✗"
        logger.info("%-30s %5.4f %8s %10s %s", exp, pp, r_min, pred, verdict)

    logger.info("\nFormula status:")
    if all_results[0]["ppl"] < 1.6: logger.info("  ✓ Dimensional form L/d_h confirmed")
    else: logger.info("  ✗ L/d_h form rejected — formula needs d_head term")
    if all_results[1]["ppl"] < 2.0: logger.info("  ✓ r_min(SmolLM2) ≤ 16 confirmed")
    else: logger.info("  ✗ r_min(SmolLM2) > 16 — η needs recalibration")
    if all_results[2]["ppl"] > 2.0: logger.info("  ✓ Below-threshold degradation confirmed at r=6")
    else: logger.info("  ✗ r=6 at plateau — r_min(SmolLM2) ≤ 6")

    return 0


if __name__ == "__main__":
    sys.exit(main())
