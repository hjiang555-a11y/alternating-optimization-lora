#!/usr/bin/env python3
"""A: SST-2 classification rank curve. Tests whether r=8 plateau extends to accuracy."""
import os, sys, json, gc, time
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=False, local_files_only=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
results = {}

ds = load_dataset("nyu-mll/glue", "sst2", split="train").select(range(500))

for rank in [4, 8, 32]:
    alpha = int(rank * 2)
    print(f">>> r={rank} a={alpha}", flush=True)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=False, local_files_only=True)
    m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=["q_proj","v_proj","k_proj","o_proj"]))
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

    def tok_fn(ex):
        t = tokenizer(ex["sentence"], truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        return {"input_ids": t["input_ids"][0], "attention_mask": t["attention_mask"][0]}
    ds_tok = ds.map(tok_fn)
    ds_tok = ds_tok.map(lambda ex: {"labels": ex["input_ids"].clone()})
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dl = DataLoader(ds_tok, batch_size=8, shuffle=True, collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]), "attention_mask": torch.stack([x["attention_mask"] for x in b]), "labels": torch.stack([x["input_ids"] for x in b])})

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=5e-5, weight_decay=0.01)
    dev = next(m.parameters()).device; m.train(); gs = 0
    while gs < 100:
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            loss = m(**b).loss / 2; loss.backward(); gs += 1
            if gs % 2 == 0: torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
            if gs >= 100: break
    elapsed = time.time() - t0
    results[f"r{rank}"] = {"rank": rank, "params_M": round(n_params/1e6, 1), "time_s": int(elapsed)}
    print(f"  Done: {n_params/1e6:.1f}M, {elapsed:.0f}s", flush=True)
    del m, base, opt; gc.collect(); torch.cuda.empty_cache()

# Eval accuracy
print("=== Accuracy ===", flush=True)
ev_ds = list(load_dataset("nyu-mll/glue", "sst2", split="validation"))
for rank in [4, 8, 32]:
    alpha = int(rank * 2)
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=False, local_files_only=True)
    m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=["q_proj","v_proj","k_proj","o_proj"]))
    correct, total = 0, 0; m.eval(); dev = next(m.parameters()).device
    with torch.no_grad():
        for ex in ev_ds:
            text = f"Sentence: {ex['sentence']}\nSentiment:"
            inp = tokenizer(text, return_tensors="pt").to(dev)
            logits = m(**inp).logits[:, -1, :]
            pos_id = tokenizer.encode("positive", add_special_tokens=False)[0]
            neg_id = tokenizer.encode("negative", add_special_tokens=False)[0]
            pred = 1 if logits[0, pos_id] > logits[0, neg_id] else 0
            correct += (pred == ex["label"]); total += 1
    acc = round(correct / max(total, 1) * 100, 1)
    results[f"r{rank}"]["accuracy"] = acc
    print(f"  r={rank}: {acc}% ({correct}/{total})", flush=True)
    del m, base; gc.collect(); torch.cuda.empty_cache()

os.makedirs("runs/a_sst2", exist_ok=True)
json.dump(results, open("runs/a_sst2/results.json", "w"), indent=2)
for k, v in results.items():
    print(f"{k}: params={v['params_M']}M, acc={v.get('accuracy', '?')}%")
