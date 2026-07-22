"""Protocol A-KD: Knowledge Distillation from ALS-optimized head.

After ALS optimizes lm_head, use the optimized head as a teacher.
The student (body + current lm_head) learns via:
  Loss = CE(student_logits, labels) + lambda * KL(teacher_logits || student_logits)

ALS-optimized head captures exact least-squares knowledge about the output
distribution. Body must learn representations that reproduce this distribution.

Compared against A-SYNC (gradient injection) and pure SGD on Qwen0.5B (24L).
"""
import json, math, time
import torch, torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer

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
def c(b):
    r = {k: torch.stack([x[k] for x in b]) for k in b[0]}
    r["labels"] = r["input_ids"].clone()
    return r
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=c)
eval_dl = DataLoader(eval_ds, batch_size=4, collate_fn=c)

def evaluate(m):
    m.eval(); tl, tn = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = m(**b)
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss): return float("inf")
            tl += out.loss.item() * b["attention_mask"].sum().item()
            tn += b["attention_mask"].sum().item()
    m.train()
    return math.exp(tl / max(tn, 1)) if tl / max(tn, 1) < 700 else float("inf")


def run_a_kd(lambda_kd=0.5, n_cycles=N_CYCLES):
    print(f"\n{'='*50}\nA-KD: lambda_kd={lambda_kd}, {n_cycles} cycles\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    lm_h = m.lm_head
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    opt = torch.optim.SGD(m.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.01, foreach=False)
    it = iter(train_dl)
    ppls, step_cnt = [], 0
    t0 = time.time()

    for cyc in range(n_cycles):
        w_before = lm_h.weight.data.cpu().clone()
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        labels = b_dev["labels"].to(device)
        step_cnt += 1

        # Phase 1: ALS on lm_head
        try: als.solve_block(b_dev, block_size=1024)
        except: pass

        # Save ALS-optimized lm_head weight
        w_als = lm_h.weight.data.clone()
        # Revert for distillation phase
        lm_h.weight.data.copy_(w_before.to(device))

        # Phase 2: KD — 2 forward passes per step
        for j in range(50):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            b2_dev = {k: v.to(device) for k, v in b2.items() if isinstance(v, torch.Tensor)}
            labels2 = b2_dev["labels"].to(device)
            opt.zero_grad()

            # Teacher forward: ALS-optimized lm_head
            lm_h.weight.data.copy_(w_als)
            with torch.no_grad():
                t_out = m(**b2_dev)
                teacher_logits = t_out.logits.detach()
            lm_h.weight.data.copy_(w_before)

            # Student forward: current lm_head
            s_out = m(**b2_dev)
            student_logits = s_out.logits

            # CE loss on student
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels2[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id or -100,
            )

            # KL loss: KL(teacher || student)
            shift_t_logits = teacher_logits[..., :-1, :].contiguous()
            shift_s_logits = student_logits[..., :-1, :].contiguous()
            kd_loss = F.kl_div(
                F.log_softmax(shift_s_logits, dim=-1),
                F.softmax(shift_t_logits, dim=-1),
                reduction="batchmean",
            )

            total_loss = ce_loss + lambda_kd * kd_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

            # Restore w_before for next student forward
            # (w_before is already on the device since opt handles it)
            # Actually we never modified lm_h in the student forward, so it's fine

        ppl = evaluate(m); ppls.append(ppl)
        print(f"  C{cyc+1:2d}: ppl={ppl:.1f}, ce={ce_loss.item():.4f}, kd={kd_loss.item():.4f}" if ppl < 1e10 else f"  C{cyc+1}: DIVERGED")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


def run_a_sync(n_cycles=N_CYCLES):
    print(f"\n{'='*50}\nA-SYNC: gradient injection, {n_cycles} cycles\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    _lm = m.lm_head
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
        print(f"  C{cyc+1:2d}: ppl={ppl:.1f}" if ppl < 1e10 else f"  C{cyc+1}: DIVERGED")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


# ── Main ──
results = {}
results["a_sync"] = run_a_sync(8)
# Try two KD strengths
results["a_kd_0.5"] = run_a_kd(lambda_kd=0.5, n_cycles=8)

print(f"\n{'='*60}")
print("Qwen0.5B (24L) — A-KD vs A-SYNC:")
for label, r in results.items():
    p = r["ppls"]
    pts = " -> ".join(f"{x:.1f}" if x < 1e10 else "inf" for x in p)
    print(f"  {label:10s}: {pts}")
    if p: print(f"             Final={p[-1]:.1f}")

with open("runs/a_kd_05b.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved runs/a_kd_05b.json")
