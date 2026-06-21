"""Quick eval of Protocol B seed 456 only — after OOM cleaned models 42+123."""
import json, torch, gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

MODEL = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=2048, padding="max_length")
tokd = ds.map(tok, batched=True, remove_columns=["text"])
tokd.set_format(type="torch", columns=["input_ids","attention_mask"])
dl = DataLoader(tokd, batch_size=1, collate_fn=lambda b: {
    "input_ids": torch.stack([x["input_ids"] for x in b]),
    "attention_mask": torch.stack([x["attention_mask"] for x in b]),
    "labels": torch.stack([x["input_ids"] for x in b]),
})

def evaluate(model, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            n = batch["attention_mask"].sum().item()
            total_loss += loss.item() * n
            total_tokens += n
            if (i+1) % 500 == 0:
                ppl = torch.exp(torch.tensor(total_loss/max(total_tokens,1))).item()
                print(f"  [{i+1}/{len(dl)}] ppl={ppl:.2f}")
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl, avg_loss, total_tokens

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
sd = torch.load("runs/qwen25_7b_800s/ckpt_Qwen25-7B_PB_800s_s456/checkpoints/step_00800/model_weights.pt", map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=False)
del sd; gc.collect(); torch.cuda.empty_cache()

device = next(model.parameters()).device
print(f"Evaluating seed 456 on device: {device}")
ppl, loss, tokens = evaluate(model, device)
print(f"RESULT: Seed 456: PPL={ppl:.2f}, loss={loss:.4f}, tokens={tokens}")

# Update the combined results
try:
    with open("runs/qwen25_7b_800s/full_test_eval.json") as f:
        results = json.load(f)
except:
    results = {}
results["B_s456"] = {"ppl": float(ppl), "loss": float(loss), "tokens": int(tokens)}
with open("runs/qwen25_7b_800s/full_test_eval.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary
b_ppls = [v['ppl'] for k,v in results.items() if k.startswith('B_s')]
if b_ppls:
    print(f"Protocol B: PPL {np.mean(b_ppls):.2f} ± {np.std(b_ppls):.2f} (N={len(b_ppls)})")
for k,v in sorted(results.items()):
    print(f"  {k}: PPL={v['ppl']:.2f}")

del model; gc.collect(); torch.cuda.empty_cache()
print("Done.")
