#!/usr/bin/env python3
"""Retry Protocol B MMLU — failed in finalize3 due to SSL transient error."""
import json, logging, sys, gc, tempfile, shutil, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retry-pb-mmlu")

MODEL_NAME = "Qwen/Qwen2.5-7B"
RUNS_DIR = Path("runs/qwen25_7b_800s")
CKPT_DIR = RUNS_DIR / "ckpt_Qwen25-7B_PB_800s_s42" / "checkpoints" / "step_00800"

def main():
    logger.info("Loading PB_s42 checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, local_files_only=True)
    sd = torch.load(str(CKPT_DIR / "model_weights.pt"), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    tmp = Path(tempfile.mkdtemp(prefix="qwen_pb42_mmlu_"))
    model.save_pretrained(str(tmp), safe_serialization=True)
    tok.save_pretrained(str(tmp))
    del model; gc.collect(); torch.cuda.empty_cache()

    logger.info("Running MMLU (5-shot, 200/task)...")
    r = simple_evaluate(model="hf", model_args={
        "pretrained": str(tmp), "trust_remote_code": False},
        tasks=["mmlu"], num_fewshot=5, limit=200, bootstrap_iters=1000)
    shutil.rmtree(tmp)

    acc = r["results"]["mmlu"].get("acc,none", -1)
    logger.info("MMLU PB_s42 acc=%.4f", acc)

    out = str(RUNS_DIR / "finalize3_mmlu_pb.json")
    with open(out, "w") as f: json.dump({"PB_s42": r}, f, indent=2, default=str)
    logger.info("Saved: %s", out)

if __name__ == "__main__":
    sys.exit(main())
