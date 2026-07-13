# Artifact Checklist — Paper v3.4

All artifacts required to reproduce "Disentangling Optimizer and Parameter Form."

## Models (HuggingFace, download on first use)

| Model | HF ID | Size | Used In |
|-------|-------|------|---------|
| GPT-2 | `gpt2` | 124M | 2×2 factorial (Table 1), depth scaling (Table 5) |
| OPT-125m | `facebook/opt-125m` | 125M | 2×2 factorial, convergence (Fig 2), overfitting (Fig 4) |
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | 494M | 2×2 factorial, rank curve, η calibration |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | 7.1B | 7B 2×2 (Table 1), downstream eval, C4 eval |
| TinyLlama-1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Depth scaling (Table 5) |
| DeepSeek-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.8B | Depth scaling (Table 5) |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` | 135M | η calibration, falsification |
| Mistral-7B | `mistralai/Mistral-7B-v0.3` | 7.2B | Depth scaling (Table 5), falsification |

## Datasets (HuggingFace `datasets`, auto-download)

| Dataset | Split | Samples | Used In |
|---------|-------|---------|---------|
| WikiText-2 (`wikitext-2-raw-v1`) | train | 128–1600 | All PPL experiments |
| WikiText-2 (`wikitext-2-raw-v1`) | test | 50–300 | All PPL evaluation |
| C4 (`c4`, `allenai/c4`, `realnewslike`) | validation | 300 | Cross-domain evaluation (§5.9) |
| HellaSwag | validation | 10,042 | Downstream eval (§5.7) |
| MMLU | test (200/subtask) | ~11,400 | Downstream eval (§5.7) |
| ARC-Challenge | test | 1,172 | Downstream eval (§5.7) |
| SST-2 | validation | 872 | Classification plateau (§5.8) |
| Chinese WikiText (`wikitext-zh`) | test | 100 | P0: language independence (§6.5) |

## Random Seeds

| Experiment | Seeds | N |
|-----------|-------|---|
| 2×2 factorial (0.5B) | 42, 123, 456 | 3 |
| 2×2 factorial (7B) | 42, 123, 456 | 3 |
| OPT-125m 800-step precision | 42, 123, 456, 789, 1024 | 5 |
| Rank curve (cross-arch) | 42 | 1 per model |
| Multi-seed rank curve (P5) | 42, 123, 456 | 3 |
| Falsification (Mistral, SmolLM2) | 42 | 1 per test |
| Downstream eval (HellaSwag) | 42, 123, 456 | 3 |

## Output Locations

| Content | Directory | Format |
|---------|-----------|--------|
| 2×2 factorial results | `runs/matrix_experiment/` | JSON |
| Multi-seed results | `runs/multi_seed_matrix/` | JSON |
| Rank curve | `runs/param_matched_baseline/`, `runs/rank_curve/` | JSON |
| Cross-architecture | `runs/cross_arch/` | JSON |
| Falsification | `runs/falsify/` | JSON |
| 7B experiments | `runs/qwen25_7b_800s/` | JSON + checkpoints |
| C4 evaluation | `runs/` (checkpoint-dependent) | JSON |
| Downstream eval | `runs/` (checkpoint-dependent) | JSON |
| η nomogram | `runs/x3_nomogram/` | JSON |
| Chinese WikiText | `runs/p0_chinese_wt/` | JSON |
| SST-2 classification | `runs/a_sst2/` | JSON |
| FFN LoRA | `runs/e4_ffn_lora/` | JSON |
| Full ASP | `runs/f2_full_asp/` | JSON |

## Figures

| Figure | Source | Script |
|--------|--------|--------|
| Fig 1: 2×2 factorial design | `figures/fig1_factorial.pdf` | Manual / matplotlib |
| Fig 2: Convergence trajectory | `figures/fig2_convergence.pdf` | `experiments/visualization.py` |
| Fig 3: Depth scaling | `figures/fig3_depth.pdf` | `experiments/visualization.py` |
| Fig 4: Overfitting | `figures/fig4_overfitting.pdf` | `experiments/visualization.py` |
| Fig 5: ALS synergy | `figures/fig5_als_synergy.pdf` | `experiments/visualization.py` |
| Fig 6: η nomogram | `figures/fig6_nomogram.pdf` | `scripts/gen_fig6_nomogram.py` |

## Hardware Requirements

| Scale | Hardware | Time (full pipeline) |
|-------|----------|---------------------|
| Small (≤1.1B) | CPU or single GPU | ~4h |
| 7B full-rank | 2× 32GB GPU + DeepSpeed ZeRO-2 | ~1h/seed |
| 7B LoRA | Single 32GB GPU | ~30min/seed |
| Full pipeline | 2× RTX 5090 (32GB) | ~12h total |

## Quick Validation (30min)

```bash
# Verify environment
pip install -r requirements.txt
python -c "import torch; import transformers; import peft; print('OK')"

# Quick test: rank curve on Qwen2.5-0.5B (r=8 only, ~5min)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', torch_dtype='auto', device_map='auto')
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj','v_proj','k_proj','o_proj']))
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
print('Environment OK')
"
```
