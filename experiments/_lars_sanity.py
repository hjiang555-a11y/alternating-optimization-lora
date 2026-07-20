"""
LARS sanity check — GPT-2 (12L), 200 steps Protocol A.
Compare LARS vs standard SGD to see if LARS digests ALS faster.
"""
import json, time, math, sys
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.framework import AltOptFramework, PhaseSchedule, PhaseConfig, Phase
from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer, LARSPhaseOptimizer
from altopt.perturbation import PerturbationScheduler

MODEL = "gpt2"
STEPS = 200
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)
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
            mask = b.get("attention_mask", torch.ones_like(b["input_ids"]))
            n = mask.sum().item()
            total_l += out.loss.item() * n
            total_n += n
    m.train()
    return {"ppl": math.exp(total_l / max(total_n, 1))}

# Baseline
baseline = evaluate(model)
print(f"Baseline PPL: {baseline['ppl']:.2f}")

# ALS config
als = ALSBlockSolver(model, reg_lambda=1e-3, step_size=0.01)
perturb = PerturbationScheduler(model, initial_scale=1e-3)

results = {}

for name, sgd_cls in [("SGD", SGDPhaseOptimizer), ("LARS", LARSPhaseOptimizer)]:
    print(f"\n{'='*50}\nTesting {name}\n{'='*50}")

    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)

    als_test = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01)
    perturb_test = PerturbationScheduler(m, initial_scale=1e-3)
    sgd_test = sgd_cls(m, lr=1e-4, momentum=0.9, weight_decay=0.01)

    schedule = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
            PhaseConfig(phase=Phase.SGD, steps=50, lr=1e-4),
            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
        ],
        cycles=4,
    )

    framework = AltOptFramework(m, schedule,
                                als_solver=als_test,
                                sgd_optimizer=sgd_test,
                                perturbation=perturb_test)

    losses = []
    ppls = []
    step = 0
    t0 = time.time()

    for cycle in range(4):
        # ALS
        als_batch = next(iter(train_dl))
        als_batch_gpu = {k: v.to(device) for k, v in als_batch.items() if isinstance(v, torch.Tensor)}
        step += 1
        l = als_test.solve_block(als_batch_gpu, block_size=1024)
        losses.append(("ALS", l))

        # SGD (50 steps via LARS/SGD)
        for _ in range(50):
            step += 1
            sgd_batch = next(iter(train_dl))
            l = sgd_test.step(sgd_batch)
            losses.append((name, l))

        # Perturb
        step += 1
        e = perturb_test.apply_noise(scale=1e-3)
        losses.append(("PERT", e))

        # Eval after each cycle
        ppl = evaluate(m)["ppl"]
        ppls.append(ppl)
        print(f"  {name} cycle {cycle+1}: step={step}, ppl={ppl:.2f}")

    elapsed = time.time() - t0
    results[name] = {"ppls": ppls, "elapsed": elapsed}
    print(f"  {name} done in {elapsed:.0f}s, final ppl={ppls[-1]:.2f}")

print(f"\n{'='*50}")
print(f"RESULTS: Baseline PPL={baseline['ppl']:.2f}")
print(f"  SGD:  {results['SGD']['ppls']}  ({results['SGD']['elapsed']:.0f}s)")
print(f"  LARS: {results['LARS']['ppls']}  ({results['LARS']['elapsed']:.0f}s)")

out = {"baseline_ppl": baseline["ppl"], "results": results}
with open("runs/lars_sanity_gpt2.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved to runs/lars_sanity_gpt2.json")
