# Experiment Configuration Reference

All hyperparameters for reproducing the 17 experiments in paper v3.4.

## Global Settings (All Experiments)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | WikiText-2 (`wikitext-2-raw-v1`) | Unless otherwise noted |
| Tokenizer | Per-model default (`AutoTokenizer`) | pad_token = eos_token |
| Max sequence length | 1024 | 2048 for 7B models |
| Batch size | 1 | Gradient accumulation = 4 (effective batch = 4) |
| Precision | bfloat16 | Float32 for Protocol A ALS phase |
| Evaluation samples | 100 | 200 for 7B (full test set: 298,938 tokens for N=3) |
| Seeds | 42, 123, 456 | N=3; N=5 for OPT-125m 800-step |
| FLOPs budget | Equalized across protocols | Per-phase costing in paper §3.3 |

## Model-Specific Hyperparameters

### Small Models (≤1.1B) — CPU / Single GPU

| Model | Layers | d_h | L/d_h | Target Modules | LR (Full/LoRA) |
|-------|--------|-----|-------|----------------|-----------------|
| GPT-2 (124M) | 12 | 768 | 0.0156 | c_attn, c_proj | 1e-4 / 1e-4 |
| OPT-125m | 12 | 768 | 0.0156 | q,v,k,o_proj | 1e-4 / 1e-4 |
| Qwen2.5-0.5B | 24 | 896 | 0.0268 | q,v,k,o_proj | 5e-5 / 5e-5 |
| TinyLlama-1.1B | 22 | 2048 | 0.0107 | q,v,k,o_proj | 1e-4 / 1e-4 |
| SmolLM2-135M | 30 | 576 | 0.0521 | q,v,k,o_proj | 1e-4 / 1e-4 |

### 7B Models — 2× RTX 5090 with DeepSpeed ZeRO-2

| Model | Layers | d_h | L/d_h | LR (Full/LoRA) | GPU Memory |
|-------|--------|-----|-------|-----------------|------------|
| Qwen2.5-7B | 28 | 3584 | 0.0078 | 5e-5 / 1e-4 | 24GB/GPU |
| Mistral-7B | 32 | 4096 | 0.0078 | 1e-4 / 1e-4 | ~24GB/GPU |
| DeepSeek-1.5B | 28 | 1536 | 0.0182 | 1e-4 / 1e-4 | Single GPU |

## Protocol-Specific Hyperparameters

### Protocol A: ASP + Full-Rank

| Parameter | Value |
|-----------|-------|
| ALS regularization λ | 1e-4 |
| ALS block size | 1024 |
| ALS step size α₀ | 0.01 |
| Depth decay β | 2.0 |
| SGD learning rate | Per-model |
| SGD momentum | 0.9 |
| SGD weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Perturbation σ₀ | 1e-3 |
| Perturbation schedule | Cosine (C_max=10) |
| Cycles | 2-4 |
| SGD steps/cycle | 50-200 |
| Numerical precision | float32 (ALS phase only) |

### Protocol B: AdamW + Full-Rank

| Parameter | Value |
|-----------|-------|
| Learning rate | Per-model |
| β₁, β₂ | 0.9, 0.999 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |

### Protocol C: ASP + LoRA

| Parameter | Value |
|-----------|-------|
| LoRA rank r | 8 |
| LoRA alpha | 16 (scaling = 2) |
| LoRA dropout | 0.0 (small models) / 0.05 (7B) |
| Target modules | Per-architecture |
| ALS (low-rank) | linalg.solve + lstsq fallback |
| B-projection λ | 1e-4 |
| SGD + Perturb | Same as Protocol A |
| Perturbation σ₀ | 5e-4 (reduced from full-rank) |

### Protocol D: AdamW + LoRA

| Parameter | Value |
|-----------|-------|
| Same LoRA config as Protocol C | |
| Same AdamW config as Protocol B | |

## Key Experiment Scripts

| Script | Purpose | Model | Time |
|--------|---------|-------|------|
| `_xval.py` | Cross-architecture rank curve | 5 models | ~30min each |
| `_falsify.py` | Falsification tests | Mistral-7B, SmolLM2 | ~20min each |
| `_finalize3.py` | Multi-seed downstream eval | Qwen2.5-7B | ~2h |
| `_param_matched_baseline.py` | Rank curve r=8-512 | Qwen2.5-0.5B | ~1h |
| `_eval_downstream.py` | HellaSwag/MMLU/ARC | Qwen2.5-7B checkpoints | ~3h |
| `_eval_c4.py` | C4 cross-domain | Qwen2.5-7B checkpoints | ~1h |
| `run_7b_gpu.py` | 7B 2×2 DeepSpeed | Qwen2.5-7B | ~1h/seed |
| `_x3_gpt2_opt.py` | η nomogram calibration | GPT-2, OPT-125m | ~10min |
| `_p0_chinese_wt.py` | Chinese WikiText | Qwen2.5-0.5B | ~15min |
| `_p4_smollm_fine.py` | SmolLM2 fine-grained | SmolLM2-135M | ~30min |
| `_e4_ffn_lora.py` | FFN LoRA | Qwen2.5-0.5B | ~20min |
| `_a_sst2.py` | SST-2 classification | Qwen2.5-0.5B | ~10min |

## Model Versions (HuggingFace)

| Model | HF ID | Notes |
|-------|-------|-------|
| GPT-2 | `gpt2` | Conv1D architecture; use `altopt.lora.LoRABaseline` not PEFT |
| OPT-125m | `facebook/opt-125m` | First nn.Linear model tested |
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | Primary 0.5B experimental model |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | Primary 7B experimental model |
| TinyLlama-1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Chat model, cross-arch validation |
| DeepSeek-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Cross-arch validation |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` | Below r_min threshold |
| Mistral-7B | `mistralai/Mistral-7B-v0.3` | Cross-arch validation |
