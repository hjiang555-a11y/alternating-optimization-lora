"""A-SYNC 8-cycle extended test on Qwen2.5-7B (28L).

Checks whether A-SYNC continues converging beyond 4 cycles or plateaus.
Clean run (no prior experiments to fragment GPU memory).
"""
import json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer
from altopt.perturbation import PerturbationScheduler

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

print(f"\nA-SYNC 8-CYCLE on Qwen2.5-7B (28L)")
print("Config: sync_strength=0.05, decay=0.8, lr=2e-4, momentum=0")
torch.cuda.empty_cache()

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")

_lm = None
for n, mod in m.named_modules():
    if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
        _lm = mod; break

als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
perturb = PerturbationScheduler(m, initial_scale=1e-3)
sgd = SGDPhaseOptimizer(m, lr=2e-4, momentum=0.0, weight_decay=0.01)
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

    ppl = evaluate(m); ppls.append(ppl)
    strength *= 0.8
    ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
    print(f"  Cycle {cyc+1}: ppl={ppl_str}, sync={strength/0.8:.4f}")

    if ppl > 1e10: break

elapsed = time.time() - t0
del m; torch.cuda.empty_cache()

result = {
    "ppls": ppls,
    "elapsed": elapsed,
    "n_cycles": N_CYCLES,
    "sync_strength": 0.05,
    "sync_decay": 0.8,
    "final_ppl": ppls[-1] if ppls else None,
    "best_ppl": min(ppls) if ppls else None,
}
print(f"\nA-SYNC 8-cycle complete: {elapsed:.0f}s")
print(f"  PPL: {' -> '.join(f'{x:.1f}' for x in ppls)}")
print(f"  Best: {min(ppls):.1f}, Final: {ppls[-1]:.1f}")

with open("runs/a_sync_8cycle_7b.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print("Saved runs/a_sync_8cycle_7b.json")
