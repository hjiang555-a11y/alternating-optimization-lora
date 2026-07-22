"""A-SYNC+EMA and A-SYNC+Warmup on Qwen2.5-7B (28L).

On Qwen0.5B all variants hit PPL 5.5 (model capacity floor).
7B is where the improvements should separate. 2 variants:

1. EMA (beta=0.3): smoothing ALS delta across cycles — reduces noise amplification
2. Warmup (4 cycles): pure SGD first, then A-SYNC — starts from a better basin

Compared against A-SYNC baseline (no-perturb, already PPL 60.9->16.6).
"""
import json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer

MODEL = "Qwen/Qwen2.5-7B"
N_CYCLES = 8
DTYPE = torch.bfloat16
device = torch.device("cuda:0")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
def tok(x): return tokenizer(x["text"], truncation=True, max_length=128, padding="max_length")
train_ds = ds["train"].map(tok, batched=True, remove_columns=["text"])
eval_ds = ds["test"].map(tok, batched=True, remove_columns=["text"])
train_ds.set_format("torch", columns=["input_ids", "attention_mask"])
eval_ds.set_format("torch", columns=["input_ids", "attention_mask"])
def collate(b):
    r = {k: torch.stack([x[k] for x in b]) for k in b[0]}
    r["labels"] = r["input_ids"].clone()
    return r
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate)
eval_dl = DataLoader(eval_ds, batch_size=2, collate_fn=collate)

def evaluate(m):
    m.eval(); total_l, total_n = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            try: out = m(**b)
            except: return float("inf")
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                return float("inf")
            total_l += out.loss.item() * b["attention_mask"].sum().item()
            total_n += b["attention_mask"].sum().item()
    m.train()
    avg = total_l / max(total_n, 1)
    return math.exp(avg) if avg < 700 else float("inf")

# ── 1. EMA ──────────────────────────────────────────────────────────

def run_ema(ema_beta=0.3, n_cycles=8):
    print(f"\n{'='*50}\nA-SYNC+EMA: beta={ema_beta}, {n_cycles} cycles\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    sgd = SGDPhaseOptimizer(m, lr=2e-4, momentum=0.0, weight_decay=0.01)
    it = iter(train_dl)
    ppls, step_cnt = [], 0
    strength = 0.05
    ema_delta = None  # CPU accumulator
    t0 = time.time()

    for cyc in range(n_cycles):
        w_before = _lm.weight.data.cpu().clone()
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        step_cnt += 1
        try: als.solve_block(b_dev, block_size=512)
        except: pass
        raw = _lm.weight.data.cpu() - w_before
        _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))

        if ema_delta is None:
            ema_delta = raw
        else:
            ema_delta = ema_beta * raw + (1 - ema_beta) * ema_delta

        for j in range(50):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            sgd.step(b2)
            if _lm.weight.grad is not None:
                g = strength * ema_delta.to(device=_lm.weight.grad.device, dtype=_lm.weight.grad.dtype)
                _lm.weight.grad.add_(g)

        ppl = evaluate(m); ppls.append(ppl)
        strength *= 0.8
        print(f"  Cycle {cyc+1}: ppl={ppl:.1f}" if ppl < 1e10 else f"  Cycle {cyc+1}: DIVERGED")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}

# ── 2. Baseline ────────────────────────────────────────────────────

def run_baseline(n_cycles=8):
    print(f"\n{'='*50}\nA-SYNC BASELINE (no-perturb), {n_cycles} cycles\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    sgd = SGDPhaseOptimizer(m, lr=2e-4, momentum=0.0, weight_decay=0.01)
    it = iter(train_dl)
    ppls, step_cnt = [], 0
    strength = 0.05
    t0 = time.time()

    for cyc in range(n_cycles):
        w_before = _lm.weight.data.cpu().clone()
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        step_cnt += 1
        try: als.solve_block(b_dev, block_size=512)
        except: pass
        delta = _lm.weight.data.cpu() - w_before
        _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))

        for j in range(50):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            sgd.step(b2)
            if _lm.weight.grad is not None:
                g = strength * delta.to(device=_lm.weight.grad.device, dtype=_lm.weight.grad.dtype)
                _lm.weight.grad.add_(g)

        ppl = evaluate(m); ppls.append(ppl)
        strength *= 0.8
        print(f"  Cycle {cyc+1}: ppl={ppl:.1f}" if ppl < 1e10 else f"  Cycle {cyc+1}: DIVERGED")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


# ── Main ────────────────────────────────────────────────────────────
results = {}
results["baseline"] = run_baseline(8)
results["ema_b0.3"] = run_ema(0.3, 8)

print(f"\n{'='*60}")
print("Qwen2.5-7B (28L) A-SYNC+EMA vs Baseline:")
for label, r in results.items():
    p = r["ppls"]
    pstr = " -> ".join(f"{x:.1f}" if x < 1e10 else "inf" for x in p)
    print(f"  {label:12s}: {pstr}")
    if p and not math.isinf(p[-1]):
        print(f"               speed/cyc: {p[1]-p[-1]:+.1f} total")

with open("runs/a_sync_plus_7b.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved runs/a_sync_plus_7b.json")
