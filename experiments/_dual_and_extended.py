"""Protocol A-DUAL: Two learning rate groups after ALS.

After ALS on lm_head, SGD with:
  - lm_head: high lr (adapts to ALS-optimal mapping fast)
  - body: low lr (keeps representations stable)

This replaces A-SYNC's gradient-injection hack with standard param groups.
Head catches up to ALS in sync with body, no residual amplification crisis.

Also extends A-SYNC to 8 cycles on 7B to check convergence asymptote.
"""
import json, math, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer
from altopt.perturbation import PerturbationScheduler

MODEL_05 = "Qwen/Qwen2.5-0.5B"
MODEL_7B = "Qwen/Qwen2.5-7B"
DTYPE_7B = torch.bfloat16

device = torch.device("cuda:0")

tokenizer_05 = AutoTokenizer.from_pretrained(MODEL_05)
tokenizer_05.pad_token = tokenizer_05.eos_token
tokenizer_7b = AutoTokenizer.from_pretrained(MODEL_7B)
tokenizer_7b.pad_token = tokenizer_7b.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1")

def build_dls(tokenizer):
    def tok(x): return tokenizer(x["text"], truncation=True, max_length=128, padding="max_length")
    tr = ds["train"].map(tok, batched=True, remove_columns=["text"])
    ev = ds["test"].map(tok, batched=True, remove_columns=["text"])
    tr.set_format("torch", columns=["input_ids", "attention_mask"])
    ev.set_format("torch", columns=["input_ids", "attention_mask"])
    def collate(b):
        r = {k: torch.stack([x[k] for x in b]) for k in b[0]}
        r["labels"] = r["input_ids"].clone()
        return r
    return (DataLoader(tr, batch_size=2, shuffle=True, collate_fn=collate),
            DataLoader(ev, batch_size=4, collate_fn=collate))

train_05, eval_05 = build_dls(tokenizer_05)
train_7b, eval_7b = build_dls(tokenizer_7b)

def evaluate(m, dl):
    m.eval(); total_l, total_n = 0.0, 0
    with torch.no_grad():
        for b in dl:
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


def run_a_dual(model_id, dtype, train_dl, eval_dl, n_cycles=4,
               head_lr=1e-3, body_lr=1e-5, label=""):
    """A-DUAL: After ALS, lm_head gets high lr, body gets low lr."""
    print(f"\n{'='*50}\nA-DUAL {label} (head_lr={head_lr}, body_lr={body_lr})\n{'='*50}")
    torch.cuda.empty_cache()

    if dtype == torch.bfloat16:
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
    else:
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)

    # Separate param groups: lm_head vs body
    head_params, body_params = [], []
    for n, p in m.named_parameters():
        if "lm_head" in n or "score" in n:
            head_params.append(p)
        else:
            body_params.append(p)

    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)

    # Manual optimizer with two param groups
    optimizer = torch.optim.SGD([
        {"params": head_params, "lr": head_lr},
        {"params": body_params, "lr": body_lr},
    ], momentum=0.0, weight_decay=0.01, foreach=False)

    it = iter(train_dl)
    ppls, step_cnt = [], 0
    t0 = time.time()

    for cyc in range(n_cycles):
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}

        # ALS
        step_cnt += 1
        try: als.solve_block(b_dev, block_size=512)
        except Exception as e: print(f"  ALS: {e}")

        # SGD with dual lr
        for j in range(50):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            optimizer.zero_grad()
            b2_dev = {k: v.to(device) for k, v in b2.items() if isinstance(v, torch.Tensor)}
            out = m(**b2_dev)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()

        step_cnt += 1
        perturb.apply_noise(scale=1e-3)
        ppl = evaluate(m, eval_dl); ppls.append(ppl)
        ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
        print(f"  Cycle {cyc+1}: ppl={ppl_str}, loss={loss.item():.4f}")

        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


def run_a_sync_extended(model_id, dtype, train_dl, eval_dl, n_cycles=8, label=""):
    """A-SYNC extended to N cycles for convergence asymptote check."""
    print(f"\n{'='*50}\nA-SYNC {label} ({n_cycles} cycles)\n{'='*50}")
    torch.cuda.empty_cache()

    if dtype == torch.bfloat16:
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
    else:
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)

    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break

    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    sgd = SGDPhaseOptimizer(m, lr=2e-4, momentum=0.0, weight_decay=0.01)
    it = iter(train_dl)

    ppls, step_cnt = [], 0
    t0 = time.time()
    strength = 0.05  # config A

    for cyc in range(n_cycles):
        w_before = _lm.weight.data.cpu().clone()
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}

        step_cnt += 1
        try: als.solve_block(b_dev, block_size=512)
        except Exception as e: print(f"  ALS: {e}")

        delta = _lm.weight.data.cpu() - w_before
        _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))

        for j in range(50):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            sgd.step(b2)
            if delta is not None and _lm.weight.grad is not None:
                g = strength * delta.to(device=_lm.weight.grad.device, dtype=_lm.weight.grad.dtype)
                _lm.weight.grad.add_(g)

        step_cnt += 1
        perturb.apply_noise(scale=1e-3)
        ppl = evaluate(m, eval_dl); ppls.append(ppl)
        strength *= 0.8
        ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
        print(f"  Cycle {cyc+1}: ppl={ppl_str}, sync={strength:.4f}")

        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


# ── Main ────────────────────────────────────────────────────────────
results_all = {}

# Experiment 1: A-DUAL on Qwen0.5B (24L)
print("\n=== EXPERIMENT 1: A-DUAL on Qwen0.5B ===")
for (hlr, blr, lbl) in [(1e-3, 1e-5, "h1e-3_b1e-5"), (1e-3, 1e-4, "h1e-3_b1e-4")]:
    results_all[f"A-DUAL-05B-{lbl}"] = run_a_dual(
        MODEL_05, torch.float32, train_05, eval_05,
        head_lr=hlr, body_lr=blr, label=f"0.5B {lbl}",
    )

# Experiment 2: A-DUAL on Qwen7B (28L)
print("\n=== EXPERIMENT 2: A-DUAL on Qwen7B ===")
results_all["A-DUAL-7B-h1e-3_b1e-5"] = run_a_dual(
    MODEL_7B, DTYPE_7B, train_7b, eval_7b,
    head_lr=1e-3, body_lr=1e-5, label="7B h1e-3_b1e-5",
)

# Experiment 3: A-SYNC extended 8-cycle on 7B
print("\n=== EXPERIMENT 3: A-SYNC 8-CYCLE on Qwen7B ===")
results_all["A-SYNC-7B-8cyc"] = run_a_sync_extended(
    MODEL_7B, DTYPE_7B, train_7b, eval_7b, n_cycles=8, label="7B x8",
)

print(f"\n{'='*60}")
print("FINAL RESULTS:")
for label, r in results_all.items():
    p = r["ppls"]
    pstr = " -> ".join(f"{x:.1f}" if x < 1e10 else "inf" for x in p)
    print(f"  {label:<30} {pstr}")

with open("runs/dual_and_extended.json", "w") as f:
    json.dump(results_all, f, indent=2, default=str)
print("Saved runs/dual_and_extended.json")
