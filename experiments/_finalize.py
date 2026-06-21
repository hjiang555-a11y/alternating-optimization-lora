#!/usr/bin/env python3
"""
Finalize: run remaining downstream + C4 evaluations sequentially.
- HellaSwag: remaining 5 checkpoints (B_s123, B_s456, D×3)
- MMLU: B+D seed 42 (5-shot, 200/task)
- ARC: B+D seed 42 (0-shot)
- C4: B+D × 3 seeds (300 samples each)

All results saved under runs/qwen25_7b_800s/finalize_*.json
"""

import json, logging, sys, time, gc, tempfile, shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finalize")

MODEL_NAME = "Qwen/Qwen2.5-7B"
RUNS_DIR = Path("runs/qwen25_7b_800s")
STEP = 800

# C4 config
C4_N_SAMPLES = 300
C4_MAX_LEN = 2048
C4_BATCH = 2


def load_model_with_checkpoint(proto, seed):
    """Load Qwen2.5-7B + checkpoint into GPU."""
    ckpt_name = f"ckpt_Qwen25-7B_P{proto}_{STEP}s_s{seed}"
    ckpt_dir = RUNS_DIR / ckpt_name / "checkpoints" / f"step_{STEP:05d}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(str(ckpt_dir))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, local_files_only=True)

    sd = torch.load(str(ckpt_dir / "model_weights.pt"), map_location="cpu")
    if proto in ("C", "D"):
        # Infer LoRA config from state_dict
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


def run_lm_eval_cmd(tasks, protocols, seeds, num_fewshot=0, limit=None, output_file=None):
    """Run lm-eval via subprocess on local checkpoint models."""
    from lm_eval import simple_evaluate

    results = {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    for proto in protocols.split(","):
        proto = proto.strip()
        for seed in [int(s.strip()) for s in seeds.split(",")]:
            label = f"P{proto}_s{seed}"
            logger.info(">>> %s: %s (fewshot=%d, limit=%s)", label, tasks, num_fewshot, limit)
            try:
                model = load_model_with_checkpoint(proto, seed)
                tmpdir = Path(tempfile.mkdtemp(prefix=f"qwen_{label}_"))
                model.save_pretrained(str(tmpdir), safe_serialization=True)
                tokenizer.save_pretrained(str(tmpdir))
                del model; gc.collect(); torch.cuda.empty_cache()

                r = simple_evaluate(model="hf", model_args={
                    "pretrained": str(tmpdir), "trust_remote_code": False,
                    "local_files_only": True},
                    tasks=tasks, num_fewshot=num_fewshot, limit=limit,
                    bootstrap_iters=1000)
                results[label] = r
                shutil.rmtree(tmpdir)

                for t, m in r.get("results", {}).items():
                    logger.info("  %s: %s", t, json.dumps({k: v for k, v in m.items()
                        if k in ("acc,none","acc_norm,none","acc_stderr,none")}, default=str))
            except Exception as e:
                logger.error("  FAIL %s: %s", label, e, exc_info=True)
                results[label] = {"error": str(e)}

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved: %s", output_file)
    return results


def run_c4_eval(output_file):
    """Evaluate C4 perplexity on B,D × 3 seeds."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Build C4 dataloader once
    logger.info("Loading C4 validation set (%d samples)...", C4_N_SAMPLES)
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        if i >= C4_N_SAMPLES: break
        texts.append(ex["text"])
    enc = tokenizer(texts, truncation=True, max_length=C4_MAX_LEN, padding="max_length", return_tensors="pt")
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"]),
                     batch_size=C4_BATCH, shuffle=False,
                     collate_fn=lambda b: {"input_ids": torch.stack([x[0] for x in b]),
                                           "attention_mask": torch.stack([x[1] for x in b]),
                                           "labels": torch.stack([x[0] for x in b])})

    def ppl(model, dl, dev):
        model.eval(); tl=0.0; tt=0
        with torch.no_grad():
            for b in dl:
                b = {k: v.to(dev) for k, v in b.items()}
                lo = model(**b).loss; nt = b["attention_mask"].sum().item()
                tl += lo.item()*nt; tt += nt
        return float(torch.exp(torch.tensor(tl/max(tt,1))).item())

    results = {}

    # Baseline
    logger.info(">>> BASELINE on C4")
    bm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                               device_map="auto", trust_remote_code=False,
                                               local_files_only=True)
    dev = next(bm.parameters()).device
    bl = ppl(bm, dl, dev)
    logger.info("Baseline C4 PPL: %.2f", bl)
    results["baseline"] = {"ppl": bl}
    del bm; gc.collect(); torch.cuda.empty_cache()

    for proto in ["B", "D"]:
        for seed in [42, 123, 456]:
            label = f"P{proto}_s{seed}"
            logger.info(">>> %s on C4", label)
            try:
                model = load_model_with_checkpoint(proto, seed)
                dev = next(model.parameters()).device
                p = ppl(model, dl, dev)
                logger.info("  C4 PPL: %.2f", p)
                results[label] = {"ppl": p, "protocol": f"P{proto}", "seed": seed}
                del model; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                logger.error("  FAIL %s: %s", label, e)
                results[label] = {"error": str(e)}

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", output_file)
    return results


def main():
    logger.info("=" * 60)
    logger.info("FINALIZE: multi-seed downstream + C4 eval")
    logger.info("=" * 60)

    # Step 1: Remaining HellaSwag (5 checkpoints)
    logger.info("\n=== STEP 1: HellaSwag remaining ===")
    run_lm_eval_cmd(
        tasks=["hellaswag"], protocols="B", seeds="123,456", num_fewshot=0,
        output_file=str(RUNS_DIR / "finalize_hellaswag_B123_B456.json"))
    run_lm_eval_cmd(
        tasks=["hellaswag"], protocols="D", seeds="42,123,456", num_fewshot=0,
        output_file=str(RUNS_DIR / "finalize_hellaswag_D.json"))

    # Step 2: MMLU
    logger.info("\n=== STEP 2: MMLU B+D s42 ===")
    run_lm_eval_cmd(
        tasks=["mmlu"], protocols="B,D", seeds="42", num_fewshot=5, limit=200,
        output_file=str(RUNS_DIR / "finalize_mmlu.json"))

    # Step 3: ARC
    logger.info("\n=== STEP 3: ARC B+D s42 ===")
    run_lm_eval_cmd(
        tasks=["arc_challenge"], protocols="B,D", seeds="42", num_fewshot=0,
        output_file=str(RUNS_DIR / "finalize_arc.json"))

    # Step 4: C4 multi-seed
    logger.info("\n=== STEP 4: C4 multi-seed ===")
    run_c4_eval(str(RUNS_DIR / "finalize_c4.json"))

    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
