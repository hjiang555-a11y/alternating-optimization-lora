#!/usr/bin/env python3
"""
Finalize v3: fixes for PEFT save + network access.
- merge_and_unload() LoRA before saving
- proxy for MMLU/ARC/C4 dataset downloads
- Runs sequentially: HellaSwag D×3 → MMLU → ARC → C4

Usage: source ~/proxy-env/load.sh && CUDA_VISIBLE_DEVICES=0 python experiments/_finalize3.py
"""

import json, logging, sys, time, gc, tempfile, shutil, os
from pathlib import Path

# Set proxy before any HF imports
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finalize3")

MODEL_NAME = "Qwen/Qwen2.5-7B"
RUNS_DIR = Path("runs/qwen25_7b_800s")
STEP = 800
C4_N, C4_LEN, C4_BS = 300, 2048, 2


def load_merged_model(proto, seed):
    """Load checkpoint, merge LoRA for Protocol C/D, return plain HF model."""
    ckpt_name = f"ckpt_Qwen25-7B_P{proto}_{STEP}s_s{seed}"
    ckpt_dir = RUNS_DIR / ckpt_name / "checkpoints" / f"step_{STEP:05d}"
    assert ckpt_dir.exists(), str(ckpt_dir)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, local_files_only=True)
    sd = torch.load(str(ckpt_dir / "model_weights.pt"), map_location="cpu")

    if proto in ("C", "D"):
        # Infer LoRA config, apply, load, merge
        targets, r = set(), None
        for k in sd:
            if "lora_A" in k:
                targets.add(k.split(".")[k.split(".").index("lora_A") - 1])
                if r is None: r = sd[k].shape[0]
        peft = get_peft_model(base, LoraConfig(
            r=r or 8, lora_alpha=16, lora_dropout=0.05,
            target_modules=sorted(targets) if targets else ["q_proj","v_proj","k_proj","o_proj"]))
        peft.load_state_dict(sd, strict=False)
        logger.info("  Merging LoRA adapter into base model...")
        merged = peft.merge_and_unload()
        del peft, base; gc.collect()
        return merged

    base.load_state_dict(sd, strict=False)
    return base


def save_model_safe(model, tokenizer, path):
    """Save model so lm-eval can load it."""
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))


def run_hellaswag_d():
    """HellaSwag on Protocol D (3 seeds)."""
    from lm_eval import simple_evaluate
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    results = {}

    for seed in [42, 123, 456]:
        label = f"PD_s{seed}"
        logger.info(">>> %s: HellaSwag", label)
        try:
            model = load_merged_model("D", seed)
            tmp = Path(tempfile.mkdtemp(prefix=f"qwen_{label}_"))
            save_model_safe(model, tok, tmp)
            del model; gc.collect(); torch.cuda.empty_cache()

            r = simple_evaluate(model="hf", model_args={
                "pretrained": str(tmp), "trust_remote_code": False},
                tasks=["hellaswag"], num_fewshot=0, bootstrap_iters=1000)
            shutil.rmtree(tmp)
            results[label] = r
            m = r["results"]["hellaswag"]
            logger.info("  HellaSwag acc=%.4f acc_norm=%.4f",
                         m.get("acc,none", -1), m.get("acc_norm,none", -1))
        except Exception as e:
            logger.error("FAIL %s: %s", label, e, exc_info=True)
            results[label] = {"error": str(e)}

    out = str(RUNS_DIR / "finalize3_hellaswag_D.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    logger.info("Saved: %s", out)


def run_task(task_name, task_id, fewshot, limit, proto_list, seeds):
    """Generic lm-eval runner for a task."""
    from lm_eval import simple_evaluate
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    all_r = {}
    for proto in proto_list:
        for seed in seeds:
            label = f"P{proto}_s{seed}"
            logger.info(">>> %s: %s (%d-shot, limit=%s)", label, task_id, fewshot, limit)
            try:
                model = load_merged_model(proto, seed)
                tmp = Path(tempfile.mkdtemp(prefix=f"qwen_{label}_"))
                save_model_safe(model, tok, tmp)
                del model; gc.collect(); torch.cuda.empty_cache()

                r = simple_evaluate(model="hf", model_args={
                    "pretrained": str(tmp), "trust_remote_code": False},
                    tasks=[task_id], num_fewshot=fewshot, limit=limit, bootstrap_iters=1000)
                shutil.rmtree(tmp)
                all_r[label] = r
                m = r["results"][task_id]
                logger.info("  %s acc=%.4f acc_norm=%.4f", task_id,
                             m.get("acc,none", -1), m.get("acc_norm,none", -1))
            except Exception as e:
                logger.error("FAIL %s: %s", label, e, exc_info=True)
                all_r[label] = {"error": str(e)}

    out = str(RUNS_DIR / f"finalize3_{task_id}.json")
    with open(out, "w") as f: json.dump(all_r, f, indent=2, default=str)
    logger.info("Saved: %s", out)
    return all_r


def run_c4():
    """C4 perplexity on B,D × 3 seeds."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    logger.info("Loading C4 validation (%d samples)...", C4_N)
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = [ex["text"] for i, ex in enumerate(ds) if i < C4_N]
    enc = tok(texts, truncation=True, max_length=C4_LEN, padding="max_length", return_tensors="pt")
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"]),
                     batch_size=C4_BS, shuffle=False,
                     collate_fn=lambda b: {
                         "input_ids": torch.stack([x[0] for x in b]),
                         "attention_mask": torch.stack([x[1] for x in b]),
                         "labels": torch.stack([x[0] for x in b])})

    def ppl(m):
        m.eval(); tl=0.0; tt=0
        dev = next(m.parameters()).device
        with torch.no_grad():
            for b in dl:
                b = {k: v.to(dev) for k, v in b.items()}
                lo = m(**b).loss; nt = b["attention_mask"].sum().item()
                tl += lo.item()*nt; tt += nt
        return round(float(torch.exp(torch.tensor(tl/max(tt,1))).item()), 2)

    results = {}
    # Baseline once
    bm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                               device_map="auto", trust_remote_code=False,
                                               local_files_only=True)
    results["baseline"] = {"ppl": ppl(bm)}
    logger.info("Baseline C4 PPL: %.2f", results["baseline"]["ppl"])
    del bm; gc.collect(); torch.cuda.empty_cache()

    for proto in ["B", "D"]:
        for seed in [42, 123, 456]:
            label = f"P{proto}_s{seed}"
            logger.info(">>> %s: C4", label)
            try:
                model = load_merged_model(proto, seed)
                p = ppl(model)
                results[label] = {"ppl": p, "protocol": f"P{proto}", "seed": seed}
                logger.info("  C4 PPL: %.2f", p)
                del model; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                logger.error("FAIL %s: %s", label, e, exc_info=True)
                results[label] = {"error": str(e)}

    out = str(RUNS_DIR / "finalize3_c4.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    logger.info("Saved: %s", out)
    return results


def main():
    logger.info("=" * 60)
    logger.info("FINALIZE v3: merged models, proxy for datasets")
    logger.info("=" * 60)

    logger.info("\n=== STEP 1: HellaSwag D x3 ===")
    run_hellaswag_d()

    logger.info("\n=== STEP 2: MMLU B,D s42 (5-shot, 200/task) ===")
    run_task("MMLU", "mmlu", 5, 200, ["B", "D"], [42])

    logger.info("\n=== STEP 3: ARC B,D s42 (0-shot) ===")
    run_task("ARC", "arc_challenge", 0, None, ["B", "D"], [42])

    logger.info("\n=== STEP 4: C4 multi-seed B,D x3 ===")
    run_c4()

    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE — check finalize3_*.json files")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
