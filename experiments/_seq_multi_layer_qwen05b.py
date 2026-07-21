"""Qwen2.5-0.5B (24L) — Protocol A with SEQUENTIAL multi-layer ALS.

Key fix: one forward pass PER LAYER, so activations never go stale.
Compares k=1 (baseline), k=4, k=8 on wikitext-2.
"""
import json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer
from altopt.perturbation import PerturbationScheduler

MODEL = "Qwen/Qwen2.5-0.5B"
N_CYCLES = 4
SGD_STEPS = 50
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

def collate(batch):
    r = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    r["labels"] = r["input_ids"].clone()
    return r

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate)
eval_dl = DataLoader(eval_ds, batch_size=4, collate_fn=collate)

def evaluate(m):
    m.eval()
    total_l, total_n = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = m(**b)
            if out.loss is None or (torch.isnan(out.loss) or torch.isinf(out.loss)):
                return float("inf")
            mask = b.get("attention_mask", torch.ones_like(b["input_ids"]))
            n = mask.sum().item()
            total_l += out.loss.item() * n
            total_n += n
    avg_loss = total_l / max(total_n, 1)
    return math.exp(avg_loss) if avg_loss < 700 else float("inf")

m0 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
baseline = evaluate(m0)
del m0; torch.cuda.empty_cache()
print(f"Baseline PPL: {baseline:.2f}")

def run(name: str, k: int) -> dict:
    print(f"\n{'='*50}\n{name} (k={k})\n{'='*50}")
    torch.cuda.empty_cache()

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, multi_layer_depth=k,
                         clip_catastrophic=10.0)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
    train_iter = iter(train_dl)

    ppls, losses, als_times = [], [], []
    step = 0
    t0 = time.time()

    for cycle in range(N_CYCLES):
        try: b = next(train_iter)
        except StopIteration: train_iter = iter(train_dl); b = next(train_iter)
        b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}

        als_t = time.time()
        step += 1
        try:
            als_loss = als.solve_block(b, block_size=1024)
        except Exception as e:
            print(f"  ALS FAILED step {step}: {e}")
            als_loss = 0.0
        als_times.append(time.time() - als_t)

        for j in range(SGD_STEPS):
            step += 1
            try: b2 = next(train_iter)
            except StopIteration: train_iter = iter(train_dl); b2 = next(train_iter)
            sgd.step(b2)

        step += 1
        perturb.apply_noise(scale=1e-3)

        ppl = evaluate(m)
        ppls.append(ppl); losses.append(als_loss)
        ppl_str = f"{ppl:.2f}" if ppl < 1e10 else "∞"
        print(f"  Cycle {cycle+1}: step={step}, ppl={ppl_str}, als_loss={als_loss:.6f}, als_time={als_times[-1]:.1f}s")
        if ppl > 1e10:
            print("  DIVERGED — stopping")
            break

    elapsed = time.time() - t0
    avg_als = sum(als_times) / max(len(als_times), 1)
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "losses": losses, "elapsed": elapsed, "als_times": als_times, "avg_als_time": avg_als, "k": k}

results = {}
for k in [1, 4, 8]:
    label = f"k{k}"
    results[label] = run(f"{'BASELINE' if k==1 else 'SEQUENTIAL'}", k=k)

print(f"\n{'='*50}")
print(f"Qwen0.5B (24L) SEQUENTIAL MULTI-LAYER ALS RESULTS: Baseline PPL={baseline:.2f}")
print(f"{'k':>6} {'ppl trajectory':>40} {'time':>8} {'als/cyc':>8}")
for label in ["k1", "k4", "k8"]:
    r = results[label]
    pstr = " → ".join(f"{x:.2f}" if x < 1e10 else "∞" for x in r["ppls"])
    print(f"  {label} {pstr:>40}  {r['elapsed']:>6.0f}s  {r['avg_als_time']:>7.1f}s")

with open("runs/seq_multi_layer_qwen05b.json", "w") as f:
    json.dump({"baseline_ppl": baseline, "results": results}, f, indent=2, default=str)
print("Saved runs/seq_multi_layer_qwen05b.json")
