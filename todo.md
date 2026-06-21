# Project Status — v1.3

**Date**: 2026-06-22
**Status**: ✅ ALL evaluations complete. Paper ready for submission.

---

## Completed Checklist

### Core Experiments
- [x] 8 architectures (GPT-2 → Qwen2.5-7B), 12L-32L
- [x] 2×2 factorial Protocol A/B/C/D on all architectures
- [x] Multi-seed (N=3-5) with PB ANOVA, Hedges' g + Bonferroni
- [x] Qwen2.5-7B Protocol B (full-rank, 3 seeds): PPL 1.25 ± 0.01
- [x] Qwen2.5-7B Protocol C (ASP-SGD on LoRA, 3 seeds): PPL 135.36 ± 9.1
- [x] Qwen2.5-7B Protocol D (LoRA r=8, 3 seeds): PPL 10.41 ± 0.01

### Parameter-Matched Baseline (§5.7)
- [x] LoRA r=256 (34.6M params): PPL 1.61/1.60/1.63 at 100/200/400 steps
- [x] LoRA r=512 (69.2M params): PPL 1.64/1.62/1.67 at 100/200/400 steps
- [x] Key finding: rank scaling 27× beats full-rank; r=256→r=512 diminishing returns

### Downstream Evaluation (§5.6.3)
- [x] HellaSwag × 3 seeds: LoRA 59.74% vs Full-rank 56.74% vs Baseline 59.91%
- [x] MMLU (5-shot): LoRA 76.34% vs Full-rank 72.16% (+4.2pp)
- [x] ARC-Challenge (0-shot): LoRA 50.43% vs Full-rank 47.18% (+3.3pp)

### Cross-Dataset Evaluation (§5.6.4)
- [x] C4 PPL × 3 seeds: LoRA 2.30 ± 0.01 vs Full-rank 2.42 ± 0.07
- [x] Key finding: WikiText 8.3× gap collapses to 1.05× on C4

### Paper Revisions
- [x] Architecture count: 9→8 (11 locations)
- [x] Phantom Appendix D: removed (2 locations)
- [x] Appendix order: A (Math), B (Figures), C (Review trace)
- [x] Table order: 4↔5 swapped
- [x] Abstract: "parameter form dominates" → parameter-count effect with caveats
- [x] Cohen's d → Hedges' g + Bonferroni correction
- [x] Depth boundary derivation caveat (ρ̄ from 2 data points)
- [x] LR scheduler, LoRA dropout, offload_optimizer documentation
- [x] Six rounds of review traceability (Appendix C)
- [x] Honest OOM disclosure for param-matched baseline
- [x] Practical Takeaways updated with rank-scaling recommendations
- [x] README updated with current findings

### Git
- [x] All changes committed and pushed to `gingersea/alternating-optimization-lora`

---

## 2×2 Factorial Matrix (Qwen2.5-7B, 800 steps)

| | ASP (ALS+SGD+Perturb) | AdamW |
|---|---|---|
| **LoRA** | C: 135.36 ± 9.1 ✅ | D: 10.41 ± 0.01 ✅ |
| **Full-rank** | A: blocked (depth boundary) ❌ | **B: 1.25 ± 0.01** ✅ |

**Updated interpretation**: The 8.3× B-vs-D WikiText gap is a triple artifact:
1. LoRA r=8 is severely underparameterized (r=256 achieves PPL=1.61 on 0.5B)
2. Full-rank overfits WikiText-2 (C4 gap collapses to 1.05×)
3. Full-rank degrades downstream performance (HellaSwag -3.2pp, MMLU -4.2pp, ARC -3.3pp)

---

## Remaining (Non-Blocking)

- [ ] C4 evaluation with full multi-seed on all protocols (currently B+D × 3 seeds, 300 samples)
- [ ] MMLU multi-seed on Protocol B (currently seed 42 only)
- [ ] >2000-step crossover verification on GPT-2/OPT-125m
- [ ] Qwen2.5-7B high-rank LoRA (r=256) experiment (GPU memory constraints)
- [ ] Submit to venue

---

*Last updated: 2026-06-22, v1.3 completion*
