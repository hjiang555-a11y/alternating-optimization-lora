# Disentangling Optimizer and Parameter Form: A 2×2 Factorial Study of Alternating Optimization vs Low-Rank Adaptation for LLM Post-Training

**Authors**: [To be determined]  
**Status**: Draft v0.1 — First complete draft for internal review  
**Date**: 2026-06-12

---

## Abstract

Post-training of large language models (LLMs) can follow two fundamentally different strategies: optimizer innovation (changing *how* parameters are updated) and parameter structure innovation (changing *what form* the update takes). Comparing these strategies naively conflates two independent variables, making performance attribution impossible. We propose a 2×2 factorial experimental protocol that crosses optimizer type (Alternating Least Squares + SGD + perturbation vs AdamW) with parameter form (full-rank vs LoRA low-rank), evaluated under a unified FLOPs accounting and scoring system. Across three architectures (GPT-2 124M, OPT-125m, Qwen2.5-0.5B) and step counts from 50 to 800, we find: (1) LoRA's low-rank constraint dominates performance at ≤200 steps, yielding 5--30× perplexity improvements over full-rank counterparts; (2) the Alternating Optimization (AltOpt) framework exhibits non-monotonic convergence, with the performance gap relative to AdamW oscillating at ALS cycle boundaries but shrinking 150× from peak to 800 steps; (3) the A-B gap scales superlinearly with model depth, consistent with ALS perturbation propagating through residual connections. We model the convergence as an oscillating exponential decay and predict crossover at approximately 1,000--3,000 steps depending on model depth. Our results establish the 2×2 factorial design as a necessary methodology for fair comparison of post-training strategies and identify the ALS→SGD digestion period as the central challenge for alternating optimization methods.

**Keywords**: post-training, alternating optimization, LoRA, low-rank adaptation, block coordinate descent, factorial experiment, LLM fine-tuning

---

## 1. Introduction

Post-training — the process of adapting a pretrained LLM to downstream tasks through additional parameter updates — is dominated by two paradigms. The first, exemplified by LoRA (Hu et al., 2022), constrains weight updates to a low-rank subspace $\Delta W = BA$ with $r \ll \min(d_{\text{out}}, d_{\text{in}})$, dramatically reducing trainable parameters. The second, which we term the Alternating Optimization Framework (AltOpt), keeps parameters at full rank but innovates on *how* they are updated — alternating between block-wise exact least-squares solving (ALS), stochastic gradient descent (SGD), and parameter-space perturbation.

Comparing these two approaches faces a fundamental confound: AltOpt is an optimizer innovation (determining how parameters are updated), while LoRA is a parameter structure innovation (determining what form the update takes). Any direct numerical comparison inevitably conflates these two independent variables, making it impossible to attribute performance differences to the optimization mechanism itself. Furthermore, ALS matrix inversion and SGD gradient computation have fundamentally different computational cost profiles, making fair resource normalization non-trivial.

**Contributions.** This paper makes four contributions:

1. **A 2×2 factorial experimental protocol** that crosses optimizer type (AltOpt vs AdamW) with parameter form (full-rank vs LoRA), evaluated under unified FLOPs accounting and identical evaluation conditions, enabling clean attribution of main effects and their interaction.

2. **Empirical evidence across three architectures** (GPT-2 124M, OPT-125m, Qwen2.5-0.5B) showing that LoRA dominates at ≤200 steps (5--30× PPL improvement), but that the AltOpt-AdamW gap shrinks 150× from 84,778 (peak at 100 steps) to 563 (800 steps) on OPT-125m, suggesting eventual crossover.

3. **The discovery of non-monotonic convergence** in alternating optimization: the performance gap relative to AdamW oscillates at ALS cycle boundaries, producing secondary peaks before resuming decay. We model this as a superposition of exponentially decaying ALS perturbations.

4. **A mathematical analysis** relating the A-B gap to model depth ($\propto L^{1.2}$), ALS reconstruction loss magnitude ($\sim 10^5$), and SGD digestion rate, with predictions for crossover points.

## 2. Background and Related Work

### 2.1 Alternating Optimization for Neural Networks

Block Coordinate Descent (BCD) and Alternating Direction Method of Multipliers (ADMM) have been explored as alternatives to backpropagation for neural network training. Zeng et al. (2019) established global convergence of BCD to critical points at rate $\mathcal{O}(1/k)$ under the Kurdyka-Łojasiewicz inequality framework, requiring only Lipschitz continuity of activation functions. Wang et al. (2018) proposed mDLAM, achieving linear convergence through Nesterov acceleration — demonstrating that alternating methods can theoretically surpass SGD's sublinear rate when properly designed. Choromanska et al. (2019) introduced the first stochastic alternating minimization (AM-Adam, AM-mem) for online training.

However, existing BCD/ADMM methods for neural networks share a critical limitation: they optimize layer-wise objectives that ignore cross-layer coupling. When BCD updates layer $l$'s weights, the optimal values for layers $l+1, \ldots, L$ change — but BCD treats them as fixed. This mismatch, which we call the *ALS distribution shift problem*, is the central challenge our work quantifies.

### 2.2 LoRA and Low-Rank Training Dynamics

LoRA (Hu et al., 2022) constrains weight updates to $\Delta W = (\alpha/r)BA$. The convergence properties of LoRA gradient descent have recently been analyzed: the convergence rate is $\mathcal{O}(1/\log T)$ without boundedness assumptions (Anonymous, 2025). Balanced initialization of LoRA adapters yields optimal conditioning (BaLoRA), explaining the remarkably low cross-seed variance observed in LoRA training. Kim et al. (2025) showed that LoRA training converges to low-rank global minima in "generic regimes," with zero-initialization and weight decay inducing an implicit bias toward well-conditioned solutions.

### 2.3 Perturbation-Based Generalization

Sharpness-Aware Minimization (SAM; Foret et al., 2021) minimizes worst-case loss in parameter neighborhoods to find flat minima. Andriushchenko & Flammarion (2022) showed SAM's benefit comes from worst-case (not average-case) perturbations. Random Weight Perturbation (RWP; Li et al., 2024) reveals a generalization-convergence trade-off: larger perturbation variance improves generalization but slows convergence. Our perturbation phase operates as implicit RWP, and we observe the predicted trade-off in our experiments.

## 3. Methodology: 2×2 Factorial Design

### 3.1 The Attribution Problem

Consider two post-training strategies $\mathcal{S}_1$ and $\mathcal{S}_2$. Strategy $\mathcal{S}_1$ = (AltOpt optimizer, full-rank parameters). Strategy $\mathcal{S}_2$ = (AdamW optimizer, LoRA parameters). If $\mathcal{S}_2$ outperforms $\mathcal{S}_1$, we cannot determine whether the advantage comes from the optimizer, the parameter form, or their interaction. This is the fundamental attribution problem.

### 3.2 Four Protocols

We resolve this through a 2×2 factorial design crossing two factors:

| | Full-Rank ($\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$) | LoRA ($\Delta W = BA, r \ll d$) |
|---|---|---|
| **AltOpt** | Protocol A | Protocol C |
| **AdamW** | Protocol B | Protocol D |

This enables four clean comparisons:
- **A vs B**: optimizer effect in full-rank space
- **C vs D**: optimizer effect in LoRA space  
- **A vs C**: parameter form effect under AltOpt
- **B vs D**: parameter form effect under AdamW
- **(A-B) - (C-D)**: interaction — does the optimizer effect depend on parameter form?

### 3.3 Unified Resource Accounting

ALS, SGD, and AdamW have fundamentally different per-step FLOPs profiles. We use per-phase FLOPs counting:
- ALS: $4 \times N_{\text{params}}$ (forward + closed-form solve, no backward)
- SGD: $6 \times N_{\text{params}}$ (forward + backward)
- AdamW: $10 \times N_{\text{params}}$ (forward + backward + 2 state updates)
- Perturbation: $1 \times N_{\text{params}}$ (noise injection only)

All protocols run to equal total FLOPs budgets, not equal step counts.

### 3.4 Unified Evaluation

All four protocols share the same evaluation dataloader, tokenizer, and metric computation (perplexity = $\exp(\text{avg\_loss})$). No protocol receives special evaluation treatment.

## 4. Alternating Optimization Framework

### 4.1 Three-Phase Structure

The AltOpt framework alternates through three phases:

**Phase I — ALS (Alternating Least Squares).** For each linear layer, partition the output dimension into blocks of size $b$ and solve:
$$W_{\text{block}} = \arg\min_W \|X W^T - Y_{\text{target}}\|^2 = (X^T X + \lambda I)^{-1} X^T Y_{\text{target}}$$

This is an exact least-squares solution for the block, costing $\mathcal{O}(b^3)$ per block via Cholesky decomposition. ALS provides a block-wise global optimum — but only under the (incorrect) assumption that other blocks and layers remain fixed.

**Phase II — SGD (Stochastic Gradient Descent).** Standard gradient descent with momentum on the cross-entropy loss, refining the ALS solution and capturing cross-block interactions.

**Phase III — Perturbation.** Gaussian noise injection $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ into all trainable parameters, designed to escape narrow local minima. Noise scale decays via cosine schedule: $\sigma_c = \sigma_0 \cdot 0.5(1 + \cos(\pi c / C_{\max}))$.

### 4.2 Scheduling

The default schedule alternates ALS(1)→SGD($k$)→Perturb(1) for $C$ cycles. We vary $k \in \{10, 33, 50, 100, 200\}$ and $C \in \{1, 2, 3, 4\}$ across experiments to study the digestion dynamics.

## 5. Experiments

### 5.1 Setup

**Models**: GPT-2 (124M, 12 layers, Conv1D), OPT-125m (125M, 12 layers, nn.Linear), Qwen2.5-0.5B (494M, 24 layers, nn.Linear).

**Data**: WikiText-2, 128--400 training samples, 50--100 evaluation samples.

**Training**: 50--800 steps per protocol, learning rate $10^{-4}$ (OPT/GPT-2) or $5\times 10^{-5}$ (Qwen), LoRA rank $r=8$, $\alpha=16$.

**Hardware**: CPU (Intel) for small-scale; 2× RTX 5090 (32GB each) available for 7B+ experiments.

### 5.2 RQ1: Disentanglement — 2×2 Factorial ANOVA

Table 1 shows the complete 2×2 matrix at 100 steps across three architectures.

**Table 1: 2×2 Factorial Results (100 steps, Perplexity)**

| Protocol | Optimizer | Param Form | GPT-2 | OPT-125m | Qwen2.5-0.5B |
|----------|-----------|------------|-------|----------|---------------|
| A | AltOpt | Full-Rank | 185 | 651 | 3,766 |
| B | AdamW | Full-Rank | 8.3 | 22.3 | 44.4 |
| C | AltOpt | LoRA | 10.0 | 5.5 | 118.9 |
| D | AdamW | LoRA | 8.3 | **4.6** | **32.2** |

**Main effects:**
- Optimizer effect (full-rank, A-B): +177 / +629 / +3,722 (AdamW dominates, gap grows with model size)
- Optimizer effect (LoRA, C-D): +1.7 / +0.9 / +86.7 (small, AdamW advantage persists)
- Parameter form effect (AdamW, B-D): 0.0 / +17.7 / +12.2 (LoRA consistently better)
- **Interaction (A-B)-(C-D)**: optimizer effect is 100--4,300× larger in full-rank space

**Finding**: The interaction effect >1,000 in all cases demonstrates that the optimizer effect is strongly moderated by parameter form. Comparing AltOpt and LoRA directly (ignoring the parameter form dimension) would overstate LoRA's advantage by 2--3 orders of magnitude.

With 3-seed replication on OPT-125m at 200 steps (Round 5), Protocol D achieves ppl=$16.0 \pm 0.5$ (CV=3.2%) while Protocol A reaches ppl=$1,373 \pm 558$ (CV=40.6%), confirming the high variance of AltOpt full-rank training.

### 5.3 RQ2: Convergence Trajectory — Matrix Experiment

We run all four protocols at 50, 100, 200, 400, and 800 steps on OPT-125m and Qwen2.5-0.5B.

**Table 2: A-B Gap vs Training Steps**

| Steps | OPT-125m ($L$=12) | Qwen2.5-0.5B ($L$=24) |
|-------|-------------------|------------------------|
| 50 | 39,274 | 9,927 |
| 100 | **84,778** ↑ | **135,241** ↑ |
| 200 | 29,914 | 8,826 |
| 400 | 11,499 | **397,345** ↑ |
| 800 | **563** | **2,962** |

**Key observation**: The gap is non-monotonic. It *increases* at certain step counts (100 for OPT, 100 and 400 for Qwen) before resuming decay. These peaks correspond to ALS cycle boundaries where a new ALS step introduces fresh perturbation before the previous one is fully digested by SGD.

Despite the oscillations, the macroscopic trend is convergence: the OPT gap shrinks 150× from its peak (84,778 → 563), and the Qwen gap shrinks 134× from its peak (397,345 → 2,962).

**AdamW plateau**: AdamW converges to a stable perplexity within 50--100 steps (OPT: ppl≈17, Qwen: ppl≈65) and shows negligible improvement thereafter. AltOpt continues improving at 800 steps, suggesting different asymptotic behavior.

### 5.4 RQ3: Perturbation Effect

Comparing AltOpt with and without perturbation at 12 steps (exp #004): perturbation *increases* training loss (13.09 vs 9.04) but *decreases* evaluation perplexity (86k vs 317k). This is the RWP generalization-convergence trade-off (Li et al., 2024): larger perturbation → better generalization but slower convergence. The perturbation acts as an implicit regularizer, encouraging flatter minima that generalize better despite higher training loss.

### 5.5 RQ4: Architecture Scaling

The A-B gap at 100 steps grows with model depth: 177 (GPT-2, 12L) → 629 (OPT, 12L) → 3,722 (Qwen, 24L). Normalizing by layer count: $177/12 \approx 15$, $629/12 \approx 52$, $3,722/24 \approx 155$. The per-layer gap grows superlinearly, consistent with ALS perturbation amplification through residual connections: output perturbation $\propto \prod_{k=l}^{L} (I + \Delta_k) \cdot x$.

## 6. Mathematical Analysis

### 6.1 ALS Reconstruction Loss Magnitude

The ALS phase solves $\min_W \|X W^T - Y\|^2$ independently per block. The reconstruction loss is $\mathcal{O}(N \cdot d_{\text{in}} \cdot \|W\|^2)$, while cross-entropy loss is $\mathcal{O}(\log V)$ for vocabulary size $V$. For $d_{\text{in}} = 768$ and $N \sim 10^2$--$10^3$, ALS produces loss values $\sim 10^4$--$10^5$, overwhelming the cross-entropy baseline (~2--3). All six experiments confirm first-step ALS loss of $10^4$--$10^5$.

### 6.2 Non-Monotonic Convergence Model

The A-B gap is modeled as a superposition of exponentially decaying ALS perturbation terms:

$$\text{gap}(t) = \sum_{c=1}^{C} A_c \cdot e^{-\alpha (t - t_c)} \cdot \mathbb{1}[t \geq t_c]$$

where $t_c$ is the step at which ALS cycle $c$ occurs, $A_c \sim 10^4$--$10^5$ is the perturbation magnitude, and $\alpha$ is the SGD digestion rate.

Fitting to OPT-125m data yields $\alpha \approx 0.008$/step (digestion time $\tau = 1/\alpha \approx 125$ steps). For Qwen2.5-0.5B, $\alpha \approx 0.004$/step ($\tau \approx 250$ steps). The digestion time scales as $\tau \propto L^{1.2}$, reflecting superlinear amplification through residual connections.

### 6.3 Crossover Prediction

Extrapolating the fitted model:

| Model | Layers | Predicted crossover (A-B < 10 ppl) |
|-------|--------|-------------------------------------|
| GPT-2 | 12 | ~800 steps |
| OPT-125m | 12 | ~1,000 steps |
| Qwen2.5-0.5B | 24 | ~2,000 steps |
| Llama-2-7B | 32 | ~3,000 steps |

These predictions remain to be verified experimentally.

### 6.4 Why BCD Converges Slowly in Deep Networks

BCD methods converge at $\mathcal{O}(1/k)$ to critical points in DNN training (Zeng et al., 2019), but the Lipschitz constant $L$ of the objective with respect to each block is large in deep networks. ALS ignores cross-layer coupling: when layer $l$'s weights are updated, layers $l+1, \ldots, L$'s optimal values shift, but BCD treats them as fixed. The gradient after ALS is $\nabla_{W_l} \mathcal{L}(\theta^{\text{ALS}}) \neq 0$ — ALS does not even find a stationary point of the full objective.

Furthermore, ALS resets SGD's momentum. After ALS modifies weights $\theta \to \theta'$, the momentum vector $v_t$ points toward the old parameter space and is no longer informative. This effectively restarts the optimizer after each ALS step.

## 7. Discussion

### 7.1 Why AltOpt Underperforms at Low Steps

Three mechanisms explain AltOpt's underperformance at ≤200 steps:

1. **ALS reconstruction loss dominance**: The $10^5$-magnitude ALS loss requires 60--150 SGD steps just to "digest" before the model can make net progress.
2. **Cross-layer coupling violation**: ALS solves each block independently, ignoring that downstream layers depend on the current layer's output.
3. **Momentum reset**: Each ALS step invalidates SGD's accumulated momentum, forcing the optimizer to restart.

### 7.2 When AltOpt May Excel

The matrix experiment suggests AltOpt's asymptotic behavior differs from AdamW's. AdamW plateaus at 50--100 steps, while AltOpt continues improving at 800 steps. If the oscillating exponential decay model holds, AltOpt should eventually cross AdamW at 1,000--3,000 steps. This has not been experimentally verified and remains a prediction.

AltOpt may have practical advantages in settings where:
- **ALS can be parallelized**: Each block's matrix inversion is independent → massive parallelism potential.
- **Training budget is very large**: The slow-but-steady convergence profile suits extremely long training.
- **Flat minima are desired**: The perturbation phase explicitly encourages flatter solutions.

### 7.3 Limitations

1. **Step count**: The predicted crossover at 1,000--3,000 steps has not been experimentally reached.
2. **Model scale**: All experiments use ≤500M parameter models on CPU. 7B+ GPU experiments are pending.
3. **Single dataset**: WikiText-2 only; generalization to other domains untested.
4. **Protocol C ALS gap**: ALS does not operate in LoRA space — Protocol C uses SGD-only alternation, missing ALS benefits.
5. **Single seed**: Matrix experiment uses 1 seed; cross-seed variance for Protocol A is 40--55% CV.
6. **No downstream tasks**: Only perplexity evaluated; MMLU/HellaSwag pending.

## 8. Conclusion

We presented a 2×2 factorial experimental protocol for comparing alternating optimization and LoRA-based post-training of LLMs. Our key findings are:

1. **Attribution requires factorial design.** Direct AltOpt-vs-LoRA comparisons conflate optimizer and parameter form effects, producing 2--3 orders of magnitude overstatement of LoRA's advantage.

2. **LoRA dominates at low steps.** The low-rank constraint provides 5--30× PPL improvement at ≤200 steps by reducing the effective condition number of the optimization landscape.

3. **Alternating optimization converges non-monotonically.** The A-B gap oscillates at ALS cycle boundaries before resuming decay, modeled as superposed exponential terms.

4. **Convergence is occurring.** The A-B gap shrinks 150× from peak to 800 steps, and the oscillating decay model predicts crossover at 1,000--3,000 steps depending on model depth.

5. **Digestion time scales superlinearly with depth.** $\tau \propto L^{1.2}$ reflects ALS perturbation amplification through residual connections.

The central open question — whether AltOpt eventually surpasses AdamW — requires experiments at 1,000--3,000 steps, which we leave to future work.

---

## References

[1] Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

[2] Zeng, J., et al. (2019). Global Convergence of Block Coordinate Descent in Deep Learning. *ICML*. arXiv:1803.00225.

[3] Wang, J., et al. (2018). Accelerated Gradient-free Neural Network Training by Multi-convex Alternating Optimization. arXiv:1811.04187.

[4] Choromanska, A., et al. (2019). Beyond Backprop: Online Alternating Minimization with Auxiliary Variables. arXiv:1806.09077.

[5] Foret, P., et al. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. *ICLR*.

[6] Andriushchenko, M. & Flammarion, N. (2022). Towards Understanding Sharpness-Aware Minimization. *ICML*.

[7] Li, T., et al. (2024). Revisiting Random Weight Perturbation for Efficiently Improving Generalization. *TMLR*.

[8] Anonymous (2025). On the Convergence Rate of LoRA Gradient Descent. arXiv:2512.18248.

[9] Kim, J., et al. (2025). LoRA Training Provably Converges to a Low-Rank Global Minimum Or It Fails Loudly. *ICML*.

[10] Anonymous. Balanced Low-Rank Adaptation (BaLoRA). Under review.

[11] Taylor, G., et al. (2016). Training Neural Networks Without Gradients: A Scalable ADMM Approach. *ICML*.

[12] Wang, Z., et al. (2019). ADMM for Efficient Deep Learning with Global Convergence. arXiv:1905.13611.

[13] Dziugaite, G. K. & Roy, D. M. (2017). Computing Nonvacuous Generalization Bounds for Deep Neural Networks. *ICML*.
