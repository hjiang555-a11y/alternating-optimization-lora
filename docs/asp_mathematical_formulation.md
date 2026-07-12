# ASP Algorithm: Mathematical Formulation

## Full-Rank and Low-Rank (LoRA) Parameterizations

This document provides the pure mathematical description of the **ASP (ALS + SGD + Perturbation)** alternating optimization algorithm, applied to both full-rank and LoRA-constrained parameter spaces. No code — only formulas, derivations, and structural analysis.

---

## 1. Problem Setting

Given a pretrained language model with parameters $\boldsymbol{\theta}_0 \in \mathbb{R}^D$ and a post-training dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, the objective is:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta} \in \Theta} \; \mathcal{L}_{\text{CE}}(\boldsymbol{\theta}) + \mathcal{R}(\boldsymbol{\theta})$$

where $\Theta$ is the admissible parameter space:
- **Full-rank**: $\Theta = \mathbb{R}^D$ — all parameters are trainable.
- **LoRA**: $\Theta = \Theta_{\text{LoRA}} \subset \mathbb{R}^D$ — only low-rank adapter matrices are trainable.

The training loss is standard causal language modeling cross-entropy:

$$\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T} \log P_{\boldsymbol{\theta}}(x_{i,t} \mid x_{i,\lt t})$$

---

## 2. The ASP Alternating Schedule

ASP alternates three phases in a fixed cycle. Let $C$ be the number of cycles and $K$ the number of SGD steps per cycle:

```
for c = 1, …, C:
    Phase I:   ALS        (1 step)
    Phase II:  SGD        (K steps)
    Phase III: Perturb    (1 step)
```

Each phase operates on the **same** parameter set but uses a fundamentally different optimization mechanism.

---

## 3. Phase I: ALS — Block-Wise Closed-Form Least Squares

### 3.1 Full-Rank ALS

For a linear layer with weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, input activations $X \in \mathbb{R}^{N \times d_{\text{in}}}$, and target output $Y \in \mathbb{R}^{N \times d_{\text{out}}}$, the ALS subproblem is:

$$W_{\text{new}} = \arg\min_{W} \|X W^\top - Y\|_F^2 + \lambda \|W\|_F^2$$

This is ridge regression with a closed-form solution:

$$\boxed{W_{\text{new}}^\top = (X^\top X + \lambda I)^{-1} X^\top Y}$$

where $\lambda = 10^{-3}$ is the regularization parameter ensuring $X^\top X + \lambda I$ is always positive-definite and thus invertible.

**Block-wise decomposition.** When $d_{\text{out}}$ is large (e.g., 151,936 for the lm_head vocabulary), the output dimension is partitioned into blocks of size $b = 1024$. For block $i$ spanning rows $[ib, (i+1)b)$:

$$\boxed{W_{\text{block}}^\top = (X^\top X + \lambda I)^{-1} X^\top Y_{\text{block}}}$$

The matrix $(X^\top X + \lambda I)^{-1}$ is computed once and reused across all blocks.

**Numerical method.** Since $X^\top X + \lambda I$ is symmetric positive-definite, it admits a Cholesky decomposition:

$$X^\top X + \lambda I = L L^\top$$

where $L \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ is lower-triangular. The solution $W_{\text{block}}^\top$ is obtained via two triangular solves rather than explicit matrix inversion:

$$L z = X^\top Y_{\text{block}}, \quad L^\top W_{\text{block}}^\top = z$$

**Target construction** differs by layer type:

- **Output layer (lm_head):** $Y_{\text{block}}$ is a one-hot encoding of the ground-truth tokens that fall within the current vocabulary block:

  $$Y_{\text{target}}[j, k] = \begin{cases} 1 & \text{if token } j \text{ has label } \text{start} + k \\ 0 & \text{otherwise} \end{cases}$$

- **Intermediate layers:** No ground-truth labels exist. ALS uses reconstruction targets:

  $$Y_{\text{block}} = X \cdot W_{\text{old,block}}^\top$$

  so the solution becomes the optimal reconstruction of the current output under $X$:

  $$W_{\text{new,block}}^\top = (X^\top X + \lambda I)^{-1} X^\top (X W_{\text{old,block}}^\top)$$

**EMA damping.** After solving, the update is damped via exponential moving average:

$$W \leftarrow (1 - \alpha(\ell)) \cdot W_{\text{old}} + \alpha(\ell) \cdot W_{\text{new}}$$

where the depth-aware step size $\alpha(\ell)$ decays exponentially with distance from the output layer:

$$\boxed{\alpha(\ell) = \alpha_0 \cdot \exp\left(-\beta \cdot \left(1 - \frac{T - 1 - \ell_{\text{idx}}}{T - 1}\right)\right)}$$

- $\ell_{\text{idx}}$: layer index (0 = nearest input)
- $T$: total number of linear layers
- $\beta = 2.0$ (depth decay coefficient)
- $\alpha_0 = 0.01$ (base step size)
- Floor: $\alpha(\ell) \geq 0.005$

### 3.2 Low-Rank ALS (for LoRA Parameterization)

When parameters are LoRA-constrained, the effective weight is:

$$W_{\text{eff}} = W_0 + \frac{\alpha}{r} \cdot B A$$

where $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is frozen (pretrained), $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ are trainable, and $\frac{\alpha}{r}$ is the LoRA scaling factor (typically $\alpha = 2r$, so scaling = 2).

**Step 1 — Full-rank ALS in effective weight space.** ALS solves for the optimal $W_{\text{new}}$ as if the weight were full-rank:

$$W_{\text{new,block}}^\top = (X^\top X + \lambda I)^{-1} X^\top (X W_{\text{eff,block}}^\top)$$

Define the discrepancy:

$$\Delta W_{\text{block}} = W_{\text{new,block}} - W_{\text{eff,block}} \quad \in \mathbb{R}^{b \times d_{\text{in}}}$$

**Step 2 — Project onto LoRA parameter space.** Since only $B$ is updated (keeping $A$ frozen during ALS), we solve:

$$\frac{\alpha}{r} \cdot \Delta B \cdot A = \Delta W_{\text{block}}$$

This is an underdetermined linear system: $\Delta B \in \mathbb{R}^{b \times r}$ and $A \in \mathbb{R}^{r \times d_{\text{in}}}$ with $r \ll d_{\text{in}}$. The minimum Frobenius-norm solution is obtained via the Moore-Penrose pseudoinverse of $A$:

$$\boxed{\Delta B = \frac{r}{\alpha} \cdot \Delta W_{\text{block}} \cdot A^\top \cdot (A A^\top + \lambda I)^{-1}}$$

**Derivation.** For the underdetermined system $\Delta B \cdot A = C$ with $C = \frac{r}{\alpha} \Delta W_{\text{block}}$:

1. The general solution is $\Delta B = C A^\dagger + Z(I - A A^\dagger)$, where $A^\dagger = A^\top (A A^\top)^{-1}$ is the Moore-Penrose pseudoinverse and $Z$ is any matrix.
2. The minimum-norm solution sets $Z = 0$, yielding $\Delta B = C A^\top (A A^\top)^{-1}$.
3. Regularizing $A A^\top$ with $\lambda I$ ensures numerical stability: $\Delta B = C A^\top (A A^\top + \lambda I)^{-1}$.

**Key efficiency property.** $A A^\top \in \mathbb{R}^{r \times r}$ is only $8 \times 8$ for $r = 8$, independent of $d_{\text{in}}$. The Cholesky factorization of this tiny matrix has negligible cost $\mathcal{O}(r^3)$.

**Update.** The B matrix is updated in-place:

$$B_{\text{new}}[i:i+b, :] = B_{\text{old}}[i:i+b, :] + \Delta B$$

---

## 4. Phase II: SGD — Gradient-Based Fine-Grained Refinement

### 4.1 Full-Rank SGD

Standard SGD with momentum and weight decay:

$$v_{t+1} = \mu \cdot v_t + g_t \quad \text{(momentum accumulation)}$$

$$W \leftarrow W - \eta \cdot v_{t+1} - \eta \lambda_{\text{wd}} \cdot W$$

where:
- $g_t = \nabla_W \mathcal{L}_{\text{CE}}$ is the gradient w.r.t. all trainable parameters
- $\eta = 10^{-4}$ (learning rate)
- $\mu = 0.9$ (momentum coefficient)
- $\lambda_{\text{wd}} = 0.01$ (weight decay)

Gradient clipping is applied: $\|g\| \leq 1.0$.

**Role of SGD in ASP.** ALS solves each layer independently, ignoring cross-layer coupling. SGD's purpose is to **coordinate** all layers: after the output layer's weights change via ALS, the representations flowing into it from earlier layers must adapt. $K = 50$–100 SGD steps provide sufficient "digestion time" for this coordination.

### 4.2 LoRA SGD

Only the LoRA adapter matrices $A, B$ receive gradients. The gradient flow through the low-rank bottleneck:

$$\nabla_A \mathcal{L} = \frac{\alpha}{r} \cdot B^\top \cdot \nabla_{h} \mathcal{L} \cdot x^\top$$

$$\nabla_B \mathcal{L} = \frac{\alpha}{r} \cdot \nabla_{h} \mathcal{L} \cdot (A x)^\top$$

where $\nabla_h \mathcal{L} \in \mathbb{R}^{d_{\text{out}}}$ is the gradient back-propagated from the output side. The Jacobian dimension is dramatically reduced:

$$\frac{\partial \mathcal{L}}{\partial (A, B)} \in \mathbb{R}^{r \times d_{\text{in}} + d_{\text{out}} \times r} \quad \text{vs.} \quad \frac{\partial \mathcal{L}}{\partial W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$

---

## 5. Phase III: Perturbation — Stochastic Noise for Local-Minima Escape

### 5.1 Noise Injection

Gaussian noise is added to all trainable parameters:

$$\theta \leftarrow \theta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_c^2 \cdot s_\ell^2)$$

where $\sigma_c$ is the cycle-dependent noise scale and $s_\ell$ is a layer-type multiplier.

### 5.2 Cosine Decay Schedule

The noise scale decays with cycle index $c$:

$$\boxed{\sigma_c = \sigma_0 \cdot \frac{1}{2}\left(1 + \cos\left(\pi \cdot \frac{c}{C_{\max}}\right)\right)}$$

- $\sigma_0 = 10^{-3}$ (full-rank) or $5 \times 10^{-4}$ (LoRA)
- $C_{\max} = 10$ (heuristic maximum cycles)
- Floor: $\sigma_c \geq 10^{-6}$

**Rationale.** Early cycles use larger perturbations to explore the loss landscape broadly (escape narrow local minima). Later cycles reduce noise to enable fine convergence within the chosen basin. This is the ASP mechanism for promoting "flat" minima — a connection to Sharpness-Aware Minimization (SAM; Foret et al., 2021).

### 5.3 Layer-Type Adaptive Scaling

Noise magnitude is scaled per layer type to reflect semantic sensitivity:

$$s_\ell = \begin{cases} 0.1 & \text{embedding layers (minimal disruption)} \\ 0.5 & \text{attention projections (moderate)} \\ 1.0 & \text{FFN/MLP layers (high redundancy)} \\ 0.5 & \text{default} \end{cases}$$

### 5.4 Perturbation in LoRA Space

When applied to LoRA parameters, the perturbation to the effective weight is:

$$(B + \varepsilon_B)(A + \varepsilon_A) - BA = \varepsilon_B A + B \varepsilon_A + \varepsilon_B \varepsilon_A$$

The perturbation in effective weight space is **always low-rank** (rank at most $2r$), meaning LoRA perturbations cannot explore the full-rank local-minimum landscape. This is an underappreciated advantage: the low-rank constraint acts as **implicit regularization**, making it harder to overfit to narrow basins.

---

## 6. Depth Boundary: ALS Perturbation Amplification

### 6.1 Residual Propagation Model

Transformer residual connections propagate ALS perturbations forward:

$$h_{\ell+1} = h_\ell + f_\ell(h_\ell; \theta_\ell)$$

Let $\delta_\ell = h_\ell^{\text{ALS}} - h_\ell$ be the perturbation introduced by ALS at layer $\ell$. Under a first-order Taylor approximation:

$$\delta_{k+1} \approx (I + J_{f_k}) \cdot \delta_k, \quad J_{f_k} = \frac{\partial f_k}{\partial h_k}$$

### 6.2 Cumulative Amplification

Iterating, the perturbation at the final layer $L$ is:

$$\boxed{\|\delta_L\| \approx \|\delta_\ell\| \cdot \prod_{k=\ell}^{L-1} \|I + J_{f_k}\|}$$

Let $\bar{\rho}$ be the geometric mean of $\|I + J_{f_k}\|$ across layers:

$$\|\delta_L\| \approx \|\delta_\ell\| \cdot \bar{\rho}^{\,L - \ell}$$

Empirically, $\bar{\rho} \approx 1.08$ (fitted from digestion-time measurements on OPT-125M and Qwen2.5-0.5B).

### 6.3 Critical Depth

SGD's recovery capacity over $T_{\text{SGD}}$ steps is:

$$C_{\text{recovery}} = \eta \cdot \mu_{\min} \cdot T_{\text{SGD}}$$

where $\mu_{\min}$ is the minimum gradient norm during recovery. The depth boundary $L_{\max}$ is the layer count at which ALS perturbation exceeds SGD recovery:

$$\boxed{L_{\max} = \frac{\ln(\eta \cdot \mu_{\min} \cdot T_{\text{SGD}} / A_{\text{eff}})}{\ln \bar{\rho}} \approx 26}$$

This predicts convergence for ≤24 layers and divergence for ≥28 layers — exactly matching all 8/8 empirical measurements.

### 6.4 Protection Mechanisms

Three protections derived from this model:

1. **Skip early layers**: ALS avoids layers with $\ell_{\text{idx}} < 0.5 \cdot T$ (longest residual chains).
2. **Depth-decay EMA**: $\alpha(\ell) \propto e^{-\beta(1 - \text{dist\_ratio})}$ — shallow layers receive exponentially smaller updates.
3. **Norm clipping**: $\frac{\|\Delta W\|_F}{\|W_{\text{old}}\|_F} \leq \tau$ — hard bound on per-layer change; catastrophic threshold triggers full rollback.

---

## 7. Computational Complexity Comparison

### 7.1 Per-Phase FLOPs

| Phase | Full-Rank | LoRA |
|-------|-----------|------|
| **ALS** (per cycle) | $N d_{\text{in}}^2 + \frac{1}{3} d_{\text{in}}^3 + 2 b d_{\text{in}}^2 \cdot n_{\text{blocks}}$ | Same as full-rank, plus $O(b \cdot r \cdot r)$ B-projection per block |
| **SGD** (per step) | $(2 + 4 + 1) \cdot D$ | $(2 + 4 + 1) \cdot r(d_{\text{in}} + d_{\text{out}})$ |
| **Perturb** (per cycle) | $D$ | $r(d_{\text{in}} + d_{\text{out}})$ |

where $D$ is the total trainable parameter count, $n_{\text{blocks}} = \lceil d_{\text{out}} / b \rceil$, and $N$ is the number of tokens in the batch.

### 7.2 Key Structural Observations

1. **ALS complexity is dominated by $X^\top X$ formation**: $\mathcal{O}(N d_{\text{in}}^2)$. This does **not** decrease under LoRA — ALS must solve in the full-rank effective-weight space before projecting back.
2. **The B-projection adds negligible cost**: $\mathcal{O}(b \cdot r \cdot r)$ per block vs. $\mathcal{O}(b \cdot d_{\text{in}}^2)$ for the main solve. For $r=8$, the projection is $\sim 10^5\times$ cheaper.
3. **SGD cost drops by $\sim 150\times$ under LoRA**: gradient computation and parameter update over $r(d_{\text{in}} + d_{\text{out}})$ parameters vs. $d_{\text{out}} d_{\text{in}}$.
4. **ALS incurs the full-rank cost under both parameterizations**: this structural mismatch — paying full-rank ALS cost while constrained to low-rank updates — is the mathematical root of the **negative synergy** observed between ASP and LoRA.

---

## 8. Unified ASP Formulations

### Protocol A: ASP + Full-Rank

$$\boxed{
\begin{aligned}
&\textbf{Input: } \boldsymbol{\theta}_0,\; \mathcal{D},\; C,\; K,\; \sigma_0 \\
&\textbf{for } c = 1, \ldots, C: \\
&\quad \text{① ALS: } W_{\text{head}}^\top = (X^\top X + \lambda I)^{-1} X^\top Y_{\text{target}} \quad \text{(output layer only)} \\
&\quad \text{② SGD: } \mathbf{v} \leftarrow \mu\mathbf{v} + \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{CE}};\; \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\mathbf{v} - \eta\lambda_{\text{wd}}\boldsymbol{\theta} \quad (K \text{ steps}) \\
&\quad \text{③ Perturb: } \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \sigma_c \cdot \boldsymbol{\varepsilon},\; \boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)
\end{aligned}}
$$

### Protocol C: ASP + LoRA (with X1 Low-Rank ALS)

$$\boxed{
\begin{aligned}
&\textbf{Input: } \{W_0^{(\ell)}\}_{\ell=1}^{L},\; \{A^{(\ell)}, B^{(\ell)}\}_{\ell=1}^{L},\; \mathcal{D},\; C,\; K,\; \sigma_0 \\
&\textbf{for } c = 1, \ldots, C: \\
&\quad \text{① Low-Rank ALS: } \\
&\qquad W_{\text{eff}} = W_0 + \frac{\alpha}{r}BA \\
&\qquad W_{\text{new}}^\top = (X^\top X + \lambda I)^{-1} X^\top (X W_{\text{eff}}^\top) \\
&\qquad \Delta B = \frac{r}{\alpha} \cdot (W_{\text{new}} - W_{\text{eff}}) \cdot A^\top \cdot (AA^\top + \lambda I)^{-1} \\
&\qquad B \leftarrow B + \Delta B \\
&\quad \text{② SGD: } A \leftarrow A - \eta\nabla_A\mathcal{L}_{\text{CE}};\; B \leftarrow B - \eta\nabla_B\mathcal{L}_{\text{CE}} \quad (K \text{ steps}) \\
&\quad \text{③ Perturb: } A \leftarrow A + \sigma_c \cdot \boldsymbol{\varepsilon}_A;\; B \leftarrow B + \sigma_c \cdot \boldsymbol{\varepsilon}_B
\end{aligned}}
$$

---

## 9. Fair Comparison: FLOPs-Normalized Protocol

Comparisons are normalized by total FLOPs, not by step count:

$$\text{All protocols run until } \sum_{t=1}^{T} \text{FLOPs}_t \geq \text{FLOPs}_{\text{budget}}$$

The per-step FLOPs accounting:

$$\text{FLOPs}_{\text{ALS}} = \underbrace{2N d_{\text{in}}^2}_{X^\top X} + \underbrace{\frac{1}{3} d_{\text{in}}^3}_{\text{Cholesky}} + \underbrace{2 n_{\text{blocks}} b d_{\text{in}}^2}_{\text{per-block triangular solves}}$$

$$\text{FLOPs}_{\text{SGD}} = (2_{\text{forward}} + 4_{\text{backward}} + 1_{\text{update}}) \cdot N_{\text{trainable}}$$

$$\text{FLOPs}_{\text{Perturb}} = N_{\text{trainable}}$$

All protocols share identical: dataloader (same batches, same shuffle seed), evaluation protocol, random seeds (N = 3–5 for multi-seed statistics), and hardware environment.

---

## 10. Summary of Key Formulae

| Formula | Description |
|---------|-------------|
| $W_{\text{new}}^\top = (X^\top X + \lambda I)^{-1} X^\top Y$ | Full-rank ALS closed-form solution |
| $\Delta B = \frac{r}{\alpha} \cdot \Delta W \cdot A^\top (AA^\top + \lambda I)^{-1}$ | Low-rank ALS B-projection |
| $\alpha(\ell) = \alpha_0 \cdot \exp(-\beta(1 - \frac{T-1-\ell_{\text{idx}}}{T-1}))$ | Depth-aware EMA damping |
| $\sigma_c = \frac{\sigma_0}{2}(1 + \cos(\pi c / C_{\max}))$ | Perturbation cosine decay |
| $\|\delta_L\| \approx \|\delta_\ell\| \cdot \bar{\rho}^{\,L-\ell}$ | Depth boundary perturbation amplification |
| $L_{\max} \approx \frac{\ln(\eta\mu_{\min}T_{\text{SGD}}/A_{\text{eff}})}{\ln\bar{\rho}} \approx 26$ | Critical depth prediction |
| $W_{\text{eff}} = W_0 + \frac{\alpha}{r}BA$ | LoRA effective weight |
| $\nabla_A \mathcal{L} = \frac{\alpha}{r} \cdot B^\top \cdot \nabla_h \mathcal{L} \cdot x^\top$ | LoRA gradient through A |
| $\nabla_B \mathcal{L} = \frac{\alpha}{r} \cdot \nabla_h \mathcal{L} \cdot (Ax)^\top$ | LoRA gradient through B |
