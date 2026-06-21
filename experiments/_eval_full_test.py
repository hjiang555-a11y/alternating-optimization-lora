"""Re-evaluate all 7B Protocol checkpoints on full WikiText-2 test set."""
import json, torch, sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

MODEL = "Qwen/Qwen2.5-7B"
BATCH = 1
MAX_LEN = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# Full test set
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
def tok(ex):
    return tokenizer(ex["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
tokd = ds.map(tok, batched=True, remove_columns=["text"])
tokd.set_format(type="torch", columns=["input_ids", "attention_mask"])
dl = DataLoader(tokd, batch_size=BATCH, collate_fn=lambda b: {
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
            if (i + 1) % 500 == 0:
                ppl = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
                print(f"  [{i+1}/{len(dl)}] ppl={ppl:.2f}, tokens={total_tokens}")
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    model.train()
    return ppl, avg_loss, total_tokens

def eval_checkpoint(ckpt_path, label, device):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Checkpoint: {ckpt_path}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        local_files_only=True)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)

    ppl, loss, tokens = evaluate(model, device)
    print(f"  RESULT: {label}: PPL={ppl:.2f}, loss={loss:.4f}, tokens={tokens}")

    del model, sd
    torch.cuda.empty_cache()
    return ppl, loss, tokens

def eval_fresh_baseline(device):
    print(f"\n{'='*60}")
    print(f"Evaluating: FRESH BASELINE (untrained Qwen2.5-7B)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        local_files_only=True)
    ppl, loss, tokens = evaluate(model, device)
    print(f"  RESULT: FRESH BASELINE: PPL={ppl:.2f}, loss={loss:.4f}, tokens={tokens}")
    del model
    torch.cuda.empty_cache()
    return ppl, loss, tokens

def main():
    device = torch.device("cuda:0")
    results = {}

    # Fresh baseline
    ppl, loss, tokens = eval_fresh_baseline(device)
    results["fresh_baseline"] = {"ppl": ppl, "loss": loss, "tokens": tokens}

    # Protocol B checkpoints (step_00800)
    for seed in [42, 123, 456]:
        ckpt = f"runs/qwen25_7b_800s/ckpt_Qwen25-7B_PB_800s_s{seed}/checkpoints/step_00800/model_weights.pt"
        if Path(ckpt).exists():
            ppl, loss, tokens = eval_checkpoint(ckpt, f"Protocol B seed {seed}", device)
            results[f"B_s{seed}"] = {"ppl": ppl, "loss": loss, "tokens": tokens}

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY (FULL TEST SET)")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: PPL={v['ppl']:.2f}, loss={v['loss']:.4f}, tokens={v['tokens']}")

    # Protocol B mean
    b_ppls = [v['ppl'] for k, v in results.items() if k.startswith('B_s')]
    if b_ppls:
        print(f"  Protocol B mean: PPL {np.mean(b_ppls):.2f} ± {np.std(b_ppls):.2f} (N={len(b_ppls)})")

    with open("runs/qwen25_7b_800s/full_test_eval.json", "w") as f:
        json.dump({k: {"ppl": float(v['ppl']), "loss": float(v['loss']), "tokens": int(v['tokens'])} for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to runs/qwen25_7b_800s/full_test_eval.json")

if __name__ == "__main__":
    main()
