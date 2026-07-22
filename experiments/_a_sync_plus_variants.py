"""A-SYNC+ — three orthogonal improvements tested on Qwen0.5B.

1. Warmup: Pure SGD pre-training before A-SYNC
2. EMA: Exponential moving average smoothing of ALS delta across cycles
3. Aligned: Only inject delta component aligned with current SGD gradient

All compared against A-SYNC baseline (no-perturb).
"""
import json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer

MODEL = "Qwen/Qwen2.5-0.5B"
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
eval_dl = DataLoader(eval_ds, batch_size=4, collate_fn=collate)

def evaluate(m):
    m.eval(); total_l, total_n = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = m(**b)
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                return float("inf")
            total_l += out.loss.item() * b["attention_mask"].sum().item()
            total_n += b["attention_mask"].sum().item()
    m.train()
    avg = total_l / max(total_n, 1)
    return math.exp(avg) if avg < 700 else float("inf")

# ── 1. Warmup ──────────────────────────────────────────────────────

def run_warmup(warmup_cycles=4, async_cycles=8):
    print(f"\n{'='*50}\nA-SYNC+WARMUP: {warmup_cycles} SGD warmup + {async_cycles} A-SYNC\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)

    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break

    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
    it = iter(train_dl)
    ppls, step_cnt = [], 0
    t0 = time.time()

    # Phase 1: Pure SGD warmup (no ALS, no sync)
    for cyc in range(warmup_cycles):
        for j in range(52):
            step_cnt += 1
            try: b = next(it)
            except StopIteration: it = iter(train_dl); b = next(it)
            sgd.step(b)
        ppl = evaluate(m); ppls.append(ppl)
        print(f"  Warmup {cyc+1}: step={step_cnt}, ppl={ppl:.1f}")

    # Phase 2: A-SYNC (no perturb)
    strength = 0.05
    for cyc in range(async_cycles):
        w_before = _lm.weight.data.cpu().clone()
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        step_cnt += 1
        try: als.solve_block(b_dev, block_size=1024)
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
        print(f"  A-SYNC {cyc+1}: step={step_cnt}, ppl={ppl:.1f}, sync={strength/0.8:.4f}")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}

# ── 2. EMA smoothing ───────────────────────────────────────────────

def run_ema(ema_beta=0.5, n_cycles=12):
    print(f"\n{'='*50}\nA-SYNC+EMA: beta={ema_beta}, {n_cycles} cycles\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break

    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
    it = iter(train_dl)
    ppls, step_cnt = [], 0
    strength = 0.05
    ema_delta = None  # smoothed delta accumulator (CPU)
    t0 = time.time()

    for cyc in range(n_cycles):
        w_before = _lm.weight.data.cpu().clone()
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        step_cnt += 1
        try: als.solve_block(b_dev, block_size=1024)
        except: pass
        raw_delta = _lm.weight.data.cpu() - w_before
        _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))

        # EMA smooth the delta
        if ema_delta is None:
            ema_delta = raw_delta
        else:
            ema_delta = ema_beta * raw_delta + (1 - ema_beta) * ema_delta

        # Use smoothed delta for injection
        delta = ema_delta

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
        print(f"  Cycle {cyc+1}: ppl={ppl:.1f}, sync={strength/0.8:.4f}")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}

# ── 3. Aligned injection ───────────────────────────────────────────

def run_aligned(n_cycles=12):
    """Only inject delta component aligned with current SGD gradient direction."""
    print(f"\n{'='*50}\nA-SYNC+ALIGNED: only aligned delta component, {n_cycles} cycles\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break

    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
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
        try: als.solve_block(b_dev, block_size=1024)
        except: pass
        delta_full = (_lm.weight.data.cpu() - w_before)
        _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))

        # Track alignment stats
        dot_sum, norm_g_sum, norm_d_sum = 0.0, 0.0, 0.0
        n_steps = 0

        for j in range(50):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            sgd.step(b2)
            if _lm.weight.grad is not None:
                grad = _lm.weight.grad.data
                delta_dev = delta_full.to(device=grad.device, dtype=grad.dtype)
                # Flatten for dot product
                grad_flat = grad.reshape(-1)
                delta_flat = delta_dev.reshape(-1)
                dot = torch.dot(grad_flat, delta_flat)
                norm_g = torch.norm(grad_flat)
                norm_d = torch.norm(delta_flat)
                if norm_g > 0 and norm_d > 0:
                    # Project delta onto gradient direction
                    proj = (dot / (norm_g * norm_g)).clamp(min=0) * grad  # only positive alignment
                    aligned_injection = proj * (strength * norm_d / (norm_g + 1e-8))
                    _lm.weight.grad.add_(aligned_injection)
                    dot_sum += dot.item(); norm_g_sum += norm_g.item(); norm_d_sum += norm_d.item()
                    n_steps += 1

        ppl = evaluate(m); ppls.append(ppl)
        strength *= 0.8
        cos_sim = dot_sum / max(norm_g_sum * norm_d_sum + 1e-8, 1e-8) if n_steps > 0 else 0
        print(f"  Cycle {cyc+1}: ppl={ppl:.1f}, cos={cos_sim:.3f}")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}

# ── 4. Baseline ────────────────────────────────────────────────────

def run_baseline(n_cycles=12):
    print(f"\n{'='*50}\nA-SYNC BASELINE (no-perturb), {n_cycles} cycles\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break

    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
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
        try: als.solve_block(b_dev, block_size=1024)
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
        print(f"  Cycle {cyc+1}: ppl={ppl:.1f}, sync={strength/0.8:.4f}")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


# ── Main ────────────────────────────────────────────────────────────
results = {}
results["baseline"] = run_baseline(12)
results["warmup"] = run_warmup(4, 8)
results["ema"] = run_ema(0.3, 12)
results["aligned"] = run_aligned(12)

print(f"\n{'='*60}")
print("A-SYNC+ VARIANTS on Qwen0.5B (24L):")
print(f"{'Variant':<15} {'PPL trajectory':>65} {'Best':>8}")
for label, r in results.items():
    p = r["ppls"]
    pstr = " -> ".join(f"{x:.1f}" if x < 1e10 else "inf" for x in p)
    best = min(float(x) for x in p if not math.isinf(x)) if p else float("inf")
    print(f"  {label:<13} {pstr:>65} {best:>8.1f}")

with open("runs/a_sync_plus_variants_05b.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved runs/a_sync_plus_variants_05b.json")
