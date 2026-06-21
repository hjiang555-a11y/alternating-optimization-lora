#!/usr/bin/env python3
"""
Finalize v2: fix PEFT model_type issue + add proxy for new datasets.
- HellaSwag D×3 (PEFT fix: copy base config.json)
- MMLU B,D s42 (with proxy for dataset download)
- ARC B,D s42 (with proxy)
- C4 multi-seed (with proxy)
"""

import json, logging, sys, time, gc, tempfile, shutil, os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finalize2")

MODEL_NAME = "Qwen/Qwen2.5-7B"
MODEL_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/"
    "snapshots/d149729398750b98c0af14eb82c78cfe92750796")
RUNS_DIR = Path("runs/qwen25_7b_800s")
STEP = 800
C4_N = 300
C4_LEN = 2048
C4_BS = 2


def load_ckpt(proto, seed):
    ckpt_name = f"ckpt_Qwen25-7B_P{proto}_{STEP}s_s{seed}"
    ckpt_dir = RUNS_DIR / ckpt_name / "checkpoints" / f"step_{STEP:05d}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(str(ckpt_dir))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CACHE, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, local_files_only=True)
    sd = torch.load(str(ckpt_dir / "model_weights.pt"), map_location="cpu")

    if proto in ("C", "D"):
        targets, r = set(), None
        for k in sd:
            if "lora_A" in k:
                parts = k.split(".")
                targets.add(parts[parts.index("lora_A") - 1])
                if r is None: r = sd[k].shape[0]
        model = get_peft_model(model, LoraConfig(
            r=r or 8, lora_alpha=16, lora_dropout=0.05,
            target_modules=sorted(targets) if targets else ["q_proj","v_proj","k_proj","o_proj"]))
    model.load_state_dict(sd, strict=False)
    return model


def save_peft_hf_model(model, save_dir):
    """Save PEFT model properly — copy base config to fix model_type."""
    model.save_pretrained(save_dir, safe_serialization=True)
    # Copy base model config.json to ensure model_type is present
    base_config = AutoConfig.from_pretrained(MODEL_CACHE, trust_remote_code=False, local_files_only=True)
    base_config.save_pretrained(save_dir)
    # Also copy tokenizer files
    for f in ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]:
        src = os.path.join(MODEL_CACHE, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(save_dir, f))


def run_hellaswag():
    """Run HellaSwag on remaining Protocol D checkpoints."""
    from lm_eval import simple_evaluate
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    results = {}
    for seed in [42, 123, 456]:
        label = f"PD_s{seed}"
        logger.info(">>> %s: HellaSwag", label)
        try:
            model = load_ckpt("D", seed)
            tmp = Path(tempfile.mkdtemp(prefix=f"qwen_{label}_"))
            save_peft_hf_model(model, str(tmp))
            tokenizer.save_pretrained(str(tmp))
            del model; gc.collect(); torch.cuda.empty_cache()

            r = simple_evaluate(model="hf", model_args={
                "pretrained": str(tmp), "trust_remote_code": False, "local_files_only": True},
                tasks=["hellaswag"], num_fewshot=0, bootstrap_iters=1000)
            results[label] = r
            shutil.rmtree(tmp)
            acc = r["results"]["hellaswag"].get("acc,none", "?")
            acc_n = r["results"]["hellaswag"].get("acc_norm,none", "?")
            logger.info("  acc=%.4f acc_norm=%.4f", acc, acc_n)
        except Exception as e:
            logger.error("FAIL %s: %s", label, e, exc_info=True)
            results[label] = {"error": str(e)}

    out = str(RUNS_DIR / "finalize2_hellaswag_D.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    logger.info("Saved: %s", out)


def run_mmlu():
    """Run MMLU on B,D s42 — needs proxy for dataset download."""
    from lm_eval import simple_evaluate
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    for proto in ["B", "D"]:
        label = f"P{proto}_s42"
        logger.info(">>> %s: MMLU (5-shot, 200/task)", label)
        try:
            model = load_ckpt(proto, 42)
            tmp = Path(tempfile.mkdtemp(prefix=f"qwen_{label}_"))
            if proto == "D":
                save_peft_hf_model(model, str(tmp))
            else:
                model.save_pretrained(str(tmp), safe_serialization=True)
            tokenizer.save_pretrained(str(tmp))
            del model; gc.collect(); torch.cuda.empty_cache()

            r = simple_evaluate(model="hf", model_args={
                "pretrained": str(tmp), "trust_remote_code": False, "local_files_only": True},
                tasks=["mmlu"], num_fewshot=5, limit=200, bootstrap_iters=1000)
            acc = r["results"]["mmlu"].get("acc,none", "?")
            logger.info("  MMLU acc=%.4f", acc)
            shutil.rmtree(tmp)
            if proto == "B": b_acc = acc
        except Exception as e:
            logger.error("FAIL %s: %s", label, e, exc_info=True)

    logger.info("MMLU: B=%.4f D=%.4f", b_acc if 'b_acc' in dir() else -1, 0)


def run_arc():
    """Run ARC-Challenge on B,D s42."""
    from lm_eval import simple_evaluate
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    for proto in ["B", "D"]:
        label = f"P{proto}_s42"
        logger.info(">>> %s: ARC-Challenge (0-shot)", label)
        try:
            model = load_ckpt(proto, 42)
            tmp = Path(tempfile.mkdtemp(prefix=f"qwen_{label}_"))
            if proto == "D":
                save_peft_hf_model(model, str(tmp))
            else:
                model.save_pretrained(str(tmp), safe_serialization=True)
            tokenizer.save_pretrained(str(tmp))
            del model; gc.collect(); torch.cuda.empty_cache()

            r = simple_evaluate(model="hf", model_args={
                "pretrained": str(tmp), "trust_remote_code": False, "local_files_only": True},
                tasks=["arc_challenge"], num_fewshot=0, bootstrap_iters=1000)
            acc = r["results"]["arc_challenge"].get("acc,none", "?")
            acc_n = r["results"]["arc_challenge"].get("acc_norm,none", "?")
            logger.info("  ARC acc=%.4f acc_norm=%.4f", acc, acc_n)
            shutil.rmtree(tmp)
        except Exception as e:
            logger.error("FAIL %s: %s", label, e, exc_info=True)


def run_c4():
    """Run C4 on B,D × 3 seeds."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading C4 validation (%d samples)...", C4_N)
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        if i >= C4_N: break
        texts.append(ex["text"])
    enc = tokenizer(texts, truncation=True, max_length=C4_LEN, padding="max_length", return_tensors="pt")
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"]),
                     batch_size=C4_BS, shuffle=False,
                     collate_fn=lambda b: {"input_ids": torch.stack([x[0] for x in b]),
                                           "attention_mask": torch.stack([x[1] for x in b]),
                                           "labels": torch.stack([x[0] for x in b])})

    def ppl(m, dl):
        m.eval(); tl=0.0; tt=0; dev=next(m.parameters()).device
        with torch.no_grad():
            for b in dl:
                b = {k: v.to(dev) for k, v in b.items()}
                lo = m(**b).loss; nt = b["attention_mask"].sum().item()
                tl += lo.item()*nt; tt += nt
        return float(torch.exp(torch.tensor(tl/max(tt,1))).item())

    results = {}
    # Baseline
    bm = AutoModelForCausalLM.from_pretrained(MODEL_CACHE, torch_dtype=torch.bfloat16,
                                               device_map="auto", trust_remote_code=False, local_files_only=True)
    bl = ppl(bm, dl)
    results["baseline"] = {"ppl": round(bl, 2)}
    logger.info("Baseline C4 PPL: %.2f", bl)
    del bm; gc.collect(); torch.cuda.empty_cache()

    for proto in ["B", "D"]:
        for seed in [42, 123, 456]:
            label = f"P{proto}_s{seed}"
            logger.info(">>> %s: C4", label)
            try:
                model = load_ckpt(proto, seed)
                p = ppl(model, dl)
                results[label] = {"ppl": round(p, 2), "protocol": f"P{proto}", "seed": seed}
                logger.info("  C4 PPL: %.2f", p)
                del model; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                logger.error("FAIL %s: %s", label, e)
                results[label] = {"error": str(e)}

    out = str(RUNS_DIR / "finalize2_c4.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    logger.info("Saved: %s", out)


def main():
    logger.info("=" * 60)
    logger.info("FINALIZE v2 $(date)")
    logger.info("=" * 60)

    logger.info("\n=== STEP 1: HellaSwag D x3 ===")
    run_hellaswag()

    logger.info("\n=== STEP 2: MMLU B,D s42 ===")
    run_mmlu()

    logger.info("\n=== STEP 3: ARC B,D s42 ===")
    run_arc()

    logger.info("\n=== STEP 4: C4 multi-seed ===")
    run_c4()

    logger.info("\nALL DONE")


if __name__ == "__main__":
    sys.exit(main())
