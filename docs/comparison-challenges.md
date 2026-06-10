# Comparison Challenges: AltOpt vs LoRA

## The Fundamental Confound

Comparing the Alternating Optimization Framework (AltOpt) with LoRA-based fine-tuning is **not an apples-to-apples comparison**. Four independent variables are simultaneously varied, making any performance difference unattributable to a single cause.

### Variable 1: Parameter Form

- **AltOpt (default)**: Updates the full-rank weight matrix directly: $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
- **LoRA**: Constrains updates to a low-rank subspace: $\Delta W = BA$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$

The low-rank constraint is both a **blessing** (fewer parameters, less memory) and a **curse** (restricted expressivity, can't represent arbitrary weight changes).

### Variable 2: Update Rule (Optimizer)

- **AltOpt**: Three-stage protocol with ALS (closed-form block solves), SGD (gradient refinement), and Perturbation (stochastic exploration)
- **LoRA (default)**: AdamW optimizer with momentum and adaptive learning rates

These optimizers have fundamentally different:
- Convergence properties (ALS is exact per block, SGD is approximate)
- Hyperparameter spaces (block size + LR + noise scale vs β₁, β₂, LR)
- Sensitivity to initialization

### Variable 3: Computational Cost (Per Step)

| Operation | AltOpt | LoRA+AdamW |
|-----------|--------|------------|
| ALS block solve | $O(N d_{\text{in}}^2 + d_{\text{in}}^3)$ per block | N/A |
| Forward pass | $O(N d)$ | $O(N d)$ |
| Backward pass | $O(N d)$ | $O(N d)$ |
| Optimizer update | SGD: $O(d)$ | AdamW: $O(d)$ (×2 for m, v states) |

**ALS is 10-100× more expensive per step than an SGD step**, but it runs rarely (once per cycle, typically 1 ALS : 100 SGD steps).

### Variable 4: Loss Landscape

- **Full-rank optimization** operates on the complete loss surface $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$
- **LoRA optimization** operates on the projected loss surface $\mathcal{L}_{\text{LoRA}}: \mathbb{R}^{r(d_{\text{out}} + d_{\text{in}})} \to \mathbb{R}$

The low-rank projection can:
1. **Smooth the landscape**: Fewer directions = fewer local minima
2. **Create artificial barriers**: Valid full-rank solutions may be unreachable in the low-rank subspace
3. **Amplify or dampen perturbation effects**: Noise added in low-rank space affects fewer directions

A critical open question: **Does LoRA's low-rank manifold make perturbation (Phase III) less effective** because there are fewer escape directions?

## Proposed Solution: 2×2 Factorial Protocol

### Design

Cross two factors in a full factorial design:

| | Full-Rank ΔW | Low-Rank ΔW (LoRA) |
|---|---|---|
| **AltOpt Optimizer** | Protocol A | Protocol C |
| **AdamW Optimizer** | Protocol B | Protocol D |

### Resource Normalization

Instead of comparing at equal steps (unfair due to different per-step costs), we compare at **equal total FLOPs**:

$$\text{Protocol runs until } \sum_{t=1}^T \text{FLOPs}_t \geq \text{BUDGET}$$

This ensures that any observed performance difference comes from *how* the budget is spent (optimizer choice), not *how much* was spent.

Three budget dimensions:
1. **FLOPs budget** (primary): Total floating-point operations
2. **Memory budget** (secondary): Peak GPU memory allocation
3. **Time budget** (tertiary): Wall-clock duration

### What Each Comparison Tests

| Comparison | Tests | Controls For |
|------------|-------|-------------|
| A vs B | Is AltOpt > AdamW for full-rank params? | Parameter form |
| C vs D | Is AltOpt > AdamW for low-rank params? | Parameter form |
| A vs C | Does full-rank help AltOpt? | Optimizer |
| B vs D | Does full-rank help AdamW? | Optimizer |
| **(A-B) - (C-D)** | Does optimizer effect depend on param form? | Interaction |

### Interpretation Guide

| Interaction | Interpretation |
|-------------|---------------|
| Not significant | Optimizer and parameter form effects are additive — AltOpt's advantage (or disadvantage) is consistent regardless of rank |
| Significant, same direction | AltOpt is better on full-rank but the gap narrows (or widens) with LoRA |
| Significant, cross-over | AltOpt is better on full-rank, AdamW is better on LoRA (or vice versa) — **most interesting case**, implies the two variables are not separable |

### Potential Outcomes

1. **AltOpt dominates** (A < B, C < D): Strong evidence for the alternating mechanism
2. **LoRA dominates** (B < A, D < C): Low-rank constraint is more impactful than optimizer
3. **Cross-over**: Optimizer choice matters only in certain parameter regimes
4. **Additive**: Both factors matter independently

## Known Confounds (Not Fully Addressed)

1. **Hyperparameter tuning**: AltOpt has different hyperparameters (block size, noise scale, phase lengths) than AdamW (β₁, β₂, ε). Incomplete tuning can bias results.
2. **Implementation efficiency**: ALS solver implementation quality (Cholesky vs iterative) affects FLOPs accounting.
3. **Model scale**: Results on GPT-2 (124M) may not transfer to 7B+ models due to different loss landscape properties.
4. **Task dependence**: Language modeling (perplexity) may favor different optimizers than instruction-following or reasoning tasks.
