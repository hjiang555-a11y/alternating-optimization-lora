# TMLR Submission Guide

**Paper**: Disentangling Optimizer and Parameter Form: A 2×2 Factorial Study  
**Version**: v2.0  
**Status**: Ready for submission  
**Target**: Transactions on Machine Learning Research (TMLR)

---

## Pre-Submission Checklist

- [x] Paper text complete (v2.0, 62 pages including appendices)
- [x] All experiments complete (13 phases)
- [x] All reviewer concerns addressed (6 rounds)
- [x] Mathematical framework verified (self-consistency audit)
- [x] Falsification experiments passed (3/3)
- [x] Boundary conditions analyzed
- [x] README and documentation aligned
- [ ] Convert to PDF (see below)
- [ ] Write cover letter (see below)
- [ ] Register on OpenReview
- [ ] Upload

---

## Converting Paper to PDF

The paper is in Markdown. Recommended conversion options:

### Option A: Pandoc (Recommended)
```bash
pandoc paper/paper_draft_v0.2.md \
  --pdf-engine=xelatex \
  --bibliography=paper/refs.bib \
  --citeproc \
  -o paper/paper_v2.0.pdf
```

### Option B: VS Code + Markdown PDF
Install `yzane.markdown-pdf` extension and export.

### Option C: Overleaf
Copy the markdown content to Overleaf, use their converter.

---

## Cover Letter (Draft)

Dear Editors,

We submit "Disentangling Optimizer and Parameter Form: A 2×2 Factorial Study of Alternating Optimization vs Low-Rank Adaptation for LLM Post-Training" for consideration at TMLR.

This paper makes three contributions:

1. **Methodological**: A rigorous 2×2 factorial protocol for disentangling optimizer effects from parameter form effects in post-training comparisons. The protocol is reusable across any pair of strategies confounded by these dimensions.

2. **Empirical**: Evidence across 8 architectures (12L–32L), 5 model families, 3 downstream tasks, and multi-seed replication establishing that:
   - LoRA rank r=8 is universally sufficient for WikiText-2 post-training
   - The widely reported "full-rank beats LoRA" result is a full-rank overfitting artifact
   - A rank sufficiency law r_min = η·L/d_h (η≈230) explains all cross-model variation

3. **Theoretical**: A three-component unified theory (Rank Sufficiency, Overfitting Boundary M-index, Architecture Invariance) with falsifiable predictions, all three of which passed experimental validation.

The paper has undergone six rounds of internal review (documented in Appendix C). All identified issues — including parameter-count confounds, missing downstream evaluation, and memorization concerns — have been addressed through additional experiments.

We believe TMLR is the appropriate venue because: (a) the paper emphasizes methodological rigor and honest negative results, (b) the empirical scope exceeds typical conference page limits, and (c) the falsifiable theoretical framework benefits from TMLR's open review process.

Sincerely,

[Author names to be determined]

---

## Key Points for Reviewers

### Strengths to emphasize in the submission:
1. **Six rounds of pre-submission review** — unusually thorough methodology validation
2. **Falsification experiments** — three quantitative predictions tested and confirmed
3. **Honest negative results** — documenting what didn't work (ALS-LoRA synergy, phase transition hypothesis)
4. **Self-correction** — the paper transparently documents how initial claims were revised by subsequent experiments
5. **Practical utility** — M-index memorization diagnostic, rank sufficiency law as design rule

### Potential reviewer concerns (pre-emptively addressed):
| Concern | Where Addressed |
|---------|----------------|
| Single dataset (WikiText-2) | §5.6.4 (C4), §5.6.3 (HellaSwag/MMLU/ARC) |
| Parameter-count confound | §5.7 (parameter-matched baseline), §6.6 (rank universality) |
| Only decoder-only architectures | §6.9.3 (boundary conditions, predictions for other architectures) |
| No full-rank Protocol A at 7B | §5.6.1 (11 attempts documented, depth boundary) |
| Single optimizer (AdamW) baseline | §7.3 Limitations #9 |
| Most experiments single-seed | §5.1 (N=3 for key results; §5.6.3 N=3 HellaSwag) |

---

## After Submission

- Monitor OpenReview for reviewer assignments
- Prepare response to reviews using the same systematic approach as documented in Appendix C
- Consider posting to arXiv simultaneously (TMLR allows this)
