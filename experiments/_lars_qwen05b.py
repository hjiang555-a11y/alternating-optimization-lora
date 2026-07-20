"""Qwen2.5-0.5B (24L) — Protocol A with LARS vs SGD. 24L is the known convergence boundary."""
import json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer, LARSPhaseOptimizer
from altopt.perturbation import PerturbationScheduler

MODEL = "Qwen/Qwen2.5-0.5B"
TRUST = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
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
        return {"ppl": float("inf")}
    avg_loss = total_l / max(total_n, 1)
    if avg_loss > 700:  # math.exp(700) ≈ 1e304, beyond float range
        return {"ppl": float("inf")}
    return {"ppl": math.exp(avg_loss)}

baseline = evaluate(model)
print(f"Baseline PPL: {baseline['ppl']:.2f}")

results = {}

for name, sgd_cls in [("SGD", SGDPhaseOptimizer), ("LARS", LARSPhaseOptimizer)]:
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    torch.cuda.empty_cache()

    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)

    if name == "LARS":
        sgd = LARSPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01, trust_coefficient=TRUST)
    else:
        sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)

    ppls = []
    step = 0
    t0 = time.time()

    for cycle in range(4):
        # ALS
        b = next(iter(train_dl))
        b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        step += 1
        als.solve_block(b, block_size=1024)

        # SGD 50
        for _ in range(50):
            step += 1
            b = next(iter(train_dl))
            sgd.step(b)

        # Perturb
        step += 1
        perturb.apply_noise(scale=1e-3)

        ppls.append(evaluate(m)["ppl"])
        print(f"  {name} cycle {cycle+1}: step={step}, ppl={ppls[-1]:.2f}")

    elapsed = time.time() - t0
    results[name] = {"ppls": ppls, "elapsed": elapsed}
    del m; torch.cuda.empty_cache()

print(f"\n{'='*50}")
print(f"Qwen2.5-0.5B (24L) RESULTS: Baseline={baseline['ppl']:.2f}")
for n in ["SGD", "LARS"]:
    r = results[n]
    print(f"  {n:4s}: {[f'{p:.2f}' for p in r['ppls']]}  ({r['elapsed']:.0f}s)")
    if len(r['ppls']) > 1:
        deltas = [f"{r['ppls'][i+1] - r['ppls'][i]:+.2f}" for i in range(len(r["ppls"]) - 1)]
        print(f"       PPL/cycle: {deltas}")

with open("runs/lars_qwen05b.json", "w") as f:
    json.dump({"baseline_ppl": baseline["ppl"], "results": results}, f, indent=2)
print("Saved to runs/lars_qwen05b.json")
