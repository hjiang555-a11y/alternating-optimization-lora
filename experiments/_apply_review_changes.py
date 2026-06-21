#!/usr/bin/env python3
"""Apply Round 4 review changes R1-R5 to paper draft."""
import re

with open('/home/room115/alternating-optimization-lora/paper/paper_draft_v0.2.md') as f:
    content = f.read()

changes = 0

# R1: Add Qwen2.5-7B row to Table 5
old = "| 7 | Mistral-7B-v0.3 | 7.2B | 32 | ✓ | **NaN** | 3,065 | diverges |"
new = old + "\n| 8 | **Qwen2.5-7B** | 7.1B | 28 | ✓ | **blocked (PPL 1.2M)** | **1.25 ± 0.01** | **depth boundary** |"
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("R1: Added Qwen2.5-7B to Table 5")

# R2: "8 architectures" → "9 architectures" everywhere
for phrase in ["8 architectures", "eight architectures", "eight architectures,"]:
    pattern = phrase
    if pattern in content:
        replacement = phrase.replace("8", "9").replace("eight", "nine").replace("Eight", "Nine")
        content = content.replace(pattern, replacement)
        changes += 1
        print(f"R2: {pattern!r} → {replacement!r}")

# Also handle "8 architectures" in Table 5 header
old_t5 = "Table 5: Architecture Scaling (8 architectures, 100-step A-B gap)"
new_t5 = "Table 5: Architecture Scaling (9 architectures, 100-step A-B gap)"
if old_t5 in content:
    content = content.replace(old_t5, new_t5)
    changes += 1

# R3: Eval caveat
old = "We report standard error of mean perplexity via 1,000 bootstrap resamples of evaluation data."
new = (old + "\n\n"
    "**7B evaluation note.** For Qwen2.5-7B experiments, the evaluation set is limited "
    "to N_EVAL=200 (~12,640 tokens) for computational efficiency. Absolute perplexity "
    "values from 7B experiments should not be compared to full WikiText-2 benchmarks; "
    "cross-protocol relative comparisons within this study remain internally valid. "
    "Full-test-set evaluation results are reported in Appendix D.")
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("R3: Added eval caveat")

# R4: Hardware + CUDA details
old = ("**Hardware**: CPU (Intel Xeon). Random seeds: 42, 123, 456 "
       "(N=3 for most experiments; N=5 for OPT-125m 800-step precision check). "
       "Code available at [repository URL].")
new = ("**Hardware**: CPU (Intel Xeon) for models ≤1.1B. Qwen2.5-7B experiments "
       "used 2× NVIDIA RTX 5090 (32GB each) with DeepSpeed ZeRO-2, "
       "DeepSpeedCPUAdam, and CPU optimizer offload (peak 24GB/GPU). "
       "The system's CUDA toolkit 12.8 differed from PyTorch's compiled CUDA 13.0; "
       "we set DS_SKIP_CUDA_CHECK=1 to bypass DeepSpeed's version assertion, "
       "which was safe because CUDA 12.8/13.0 driver APIs are compatible for "
       "NCCL collectives (protocol B only needed NCCL all-reduce, not CUDA JIT). "
       "Random seeds: 42, 123, 456 (N=3 for most experiments; N=5 for OPT-125m "
       "800-step precision check). Code available at [repository URL].")
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("R4: Added hardware/CUDA details")

# R5: FSDP memory diagnostic
old = "Protocol A diverged on both GPU models, confirming the depth boundary"
new = ("Protocol A diverged on all GPU models. On Qwen2.5-7B FSDP "
       "(FULL_SHARD + CPU offload, per-layer auto_wrap_policy), both GPUs "
       "maintained stable 30.2/32GB memory for the full 704-step training run, "
       "confirming the failure was algorithmic (ALS perturbation amplification "
       "through 28 residual layers) rather than hardware memory exhaustion")
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("R5: Added FSDP memory diagnostic")

# Update contribution 4 text
old_c4 = "across 8 architectures, including GPU validation at 7B scale"
new_c4 = "across 9 architectures, including exhaustive GPU validation at 7B scale (11 attempts, 2 backends)"
if old_c4 in content:
    content = content.replace(old_c4, new_c4)
    changes += 1

# Update line about "8 architectures" in body
old_body = "across eight architectures including GPU-trained models"
if old_body in content:
    content = content.replace(old_body,
        "across nine architectures including GPU-trained models at 7B scale")
    changes += 1

with open('/home/room115/alternating-optimization-lora/paper/paper_draft_v0.2.md', 'w') as f:
    f.write(content)

print(f"\nTotal: {changes} changes applied")
