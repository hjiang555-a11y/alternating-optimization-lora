"""A-SYNC 8-cycle on Qwen0.5B (24L) + no-perturb ablation.

Q1: Does cycle-4 regression (8.4→46.1) self-correct in cycles 5-8?
Q2: Can we drop the perturb phase entirely? (A-SYNC clean protocol)

Two runs: standard (+perturb) and clean (no perturb).
"""
import json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer
from altopt.perturbation import PerturbationScheduler

MODEL = "Qwen/Qwen2.5-0.5B"
N_CYCLES = 8
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

def run(use_perturb, label):
    print(f"\n{'='*50}\nA-SYNC 8-cycle 0.5B: {label}\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)

    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break

    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    perturb = PerturbationScheduler(m, initial_scale=1e-3) if use_perturb else None
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
    it = iter(train_dl)

    ppls, step_cnt = [], 0
    strength = 0.05
    t0 = time.time()

    for cyc in range(N_CYCLES):
        w_before = _lm.weight.data.cpu().clone()
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}

        step_cnt += 1
        try: als.solve_block(b_dev, block_size=1024)
        except: pass

        delta = (_lm.weight.data.cpu() - w_before).cpu()
        _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))

        for j in range(50):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            sgd.step(b2)
            if _lm.weight.grad is not None:
                g = strength * delta.to(device=_lm.weight.grad.device, dtype=_lm.weight.grad.dtype)
                _lm.weight.grad.add_(g)

        if perturb:
            step_cnt += 1
            perturb.apply_noise(scale=1e-3)

        ppl = evaluate(m); ppls.append(ppl)
        strength *= 0.8
        ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
        print(f"  Cycle {cyc+1}: ppl={ppl_str}, sync={strength/0.8:.4f}")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}

# Run both
results = {}
results["with_perturb"] = run(True, "+perturb")
results["no_perturb"] = run(False, "NO perturb")

print(f"\n{'='*50}")
print("Qwen0.5B A-SYNC 8-cycle:")
for label, r in results.items():
    p = r["ppls"]
    pstr = " -> ".join(f"{x:.1f}" if x < 1e10 else "inf" for x in p)
    best = min(p) if p else float("inf")
    print(f"  {label:15s}: {pstr}")
    print(f"              Best: {best:.1f}")

with open("runs/a_sync_8cycle_qwen05b.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved runs/a_sync_8cycle_qwen05b.json")
