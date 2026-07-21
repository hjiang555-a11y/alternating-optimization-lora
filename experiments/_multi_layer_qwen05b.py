"""Qwen2.5-0.5B (24L) — Protocol A with multi-layer ALS k=1 (baseline) vs k=8.

Tests whether solving last K transformer blocks via ALS stabilizes the
residual amplification feedback loop in deep models.
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
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
    ok = True
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = m(**b)
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                ok = False; break
            mask = b.get("attention_mask", torch.ones_like(b["input_ids"]))
            n = mask.sum().item()
            total_l += out.loss.item() * n
            total_n += n
    m.train()
    if not ok or total_n == 0:
        return float("inf")
    avg_loss = total_l / max(total_n, 1)
    if avg_loss > 700:
        return float("inf")
    return math.exp(avg_loss)

baseline = evaluate(model)
print(f"Baseline PPL: {baseline:.2f}")

def run(name: str, k: int) -> dict:
    print(f"\n{'='*50}\n{name} (k={k})\n{'='*50}")
    torch.cuda.empty_cache()

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    als = ALSBlockSolver(
        m, reg_lambda=1e-3, step_size=0.01,
        multi_layer_depth=k,
        clip_catastrophic=10.0,
    )
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)

    train_iter = iter(train_dl)

    ppls = []
    losses = []
    step = 0
    t0 = time.time()

    for cycle in range(N_CYCLES):
        try:
            b = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            b = next(train_iter)

        b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}

        # ALS
        step += 1
        try:
            als_loss = als.solve_block(b, block_size=1024)
        except Exception as e:
            print(f"  ALS failed step {step}: {e}")
            als_loss = 0.0

        # SGD
        for j in range(SGD_STEPS):
            step += 1
            try:
                b2 = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                b2 = next(train_iter)
            sgd.step(b2)

        # Perturb
        step += 1
        perturb.apply_noise(scale=1e-3)

        ppl = evaluate(m)
        ppls.append(ppl)
        losses.append(als_loss)
        print(f"  Cycle {cycle+1}: step={step}, ppl={ppl:.2f}, als_loss={als_loss:.6f}")

        if ppl > 1e10:
            print("  DIVERGED — stopping")
            break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "losses": losses, "elapsed": elapsed, "k": k}

results = {}

# k=1 baseline
results["k1"] = run("BASELINE k=1", k=1)

# k=8 multi-layer
results["k8"] = run("MULTI-LAYER k=8", k=8)

print(f"\n{'='*50}")
print(f"Qwen0.5B (24L) RESULTS: Baseline PPL={baseline:.2f}")
for label in ["k1", "k8"]:
    r = results[label]
    p = r["ppls"]
    if p:
        pstr = [f"{x:.2f}" if x < 1e10 else "∞" for x in p]
        print(f"  {label}: {pstr}  ({r['elapsed']:.0f}s)  k={r['k']}")
        if len(p) > 1 and all(x < 1e10 for x in p):
            deltas = [f"{p[i+1] - p[i]:+.2f}" for i in range(len(p) - 1)]
            print(f"       PPL/cycle: {deltas}")

out = {"baseline_ppl": baseline, "results": results}
with open("runs/multi_layer_qwen05b.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("Saved to runs/multi_layer_qwen05b.json")
