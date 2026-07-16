# What Problem Does This Paper Solve?

## A statement of the research question, with mathematical formulation

---

## 1. The Setup

We begin with a pretrained autoregressive language model parameterized by $\boldsymbol{\theta}_0 \in \mathbb{R}^D$, where $D$ is the total parameter count ($D \approx 5 \times 10^8$ for a 0.5B model, $\approx 7 \times 10^9$ for a 7B model). The model defines a conditional distribution $P_{\boldsymbol{\theta}}(y_t \mid y_{<t}, x)$ over token sequences.

**Post-training** (fine-tuning) seeks parameters $\boldsymbol{\theta}^*$ that minimize the cross-entropy loss on a task-specific dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, where typically $N \sim 10^2$–$10^3$ text samples:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta} \in \Theta} \; \mathcal{L}_{\text{CE}}(\boldsymbol{\theta}; \mathcal{D}), \qquad \mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i} \log P_{\boldsymbol{\theta}}(y_{i,t} \mid y_{i,<t}, x_i)$$

The central tension: $D \gg N$ by 3–6 orders of magnitude. The model has vastly more parameters than training examples. How we constrain the parameter update determines whether the model generalizes or memorizes.

---

## 2. Two Independent Design Dimensions

Post-training involves two choices that the literature routinely conflates:

### Dimension 1: Parameter Form — *what* gets updated?

**Full-rank** ($\Theta_{\text{full}} = \mathbb{R}^D$). Every parameter may change. For a linear layer with pretrained weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:

$$\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \quad \text{(unconstrained)}$$

The number of trainable parameters equals the full model size: $N_p = D \approx 5 \times 10^8$.

**LoRA** ($\Theta_{\text{LoRA}} \subset \mathbb{R}^D$). The update is constrained to rank $r \ll \min(d_{\text{out}}, d_{\text{in}})$:

$$W_{\text{eff}} = W_0 + \frac{\alpha}{r} \cdot B A, \qquad A \in \mathbb{R}^{r \times d_{\text{in}}},\; B \in \mathbb{R}^{d_{\text{out}} \times r}$$

$W_0$ is frozen; only $A$ and $B$ are trainable. For $r = 8$, $N_p \approx 4r d_h L \approx 1.4 \times 10^6$ for a 0.5B model—a reduction of $\sim 350\times$.

### Dimension 2: Optimizer — *how* are parameters updated?

**AdamW** (standard). Adaptive first-order method maintaining per-parameter momentum $m_t$ and second-moment estimates $v_t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda_{\text{wd}} \boldsymbol{\theta}_t$$

**ASP** (our proposed alternative). Three-phase alternating cycle:

1. **ALS**: For the output layer, solve the exact regularized least-squares problem:

   $$W_{\text{new}}^\top = (X^\top X + \lambda I)^{-1} X^\top Y_{\text{target}}$$

   This is the **closed-form global minimizer** of $\|XW^\top - Y\|_F^2 + \lambda\|W\|_F^2$—no iteration, no learning rate, no gradient noise.

2. **SGD**: Gradient descent with momentum ($\mu$) and weight decay ($\lambda_{\text{wd}}$) to coordinate all layers after ALS perturbs hidden representations:

   $$v \leftarrow \mu v + \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{CE}}, \qquad \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta v - \eta \lambda_{\text{wd}} \boldsymbol{\theta}$$

3. **Perturbation**: Gaussian noise with cosine decay to escape narrow local minima:

   $$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_c^2), \quad \sigma_c = \frac{\sigma_0}{2}\left(1 + \cos\frac{\pi c}{C_{\max}}\right)$$

---

## 3. The $2 \times 2$ Factorial Design

Crossing the two dimensions yields four protocols:

|  | Full-rank ($\Theta_{\text{full}}$) | LoRA ($\Theta_{\text{LoRA}}$) |
|--|-------------------------------------|-------------------------------|
| **ASP** | Protocol A | Protocol C |
| **AdamW** | Protocol B | Protocol D |

The factorial design decomposes the outcome (e.g., perplexity $\mu_{ij}$) into:

$$\begin{aligned}
\text{Optimizer main effect:} \quad & \text{ME}_{\text{opt}} = \frac{(\mu_{\text{A}} + \mu_{\text{C}}) - (\mu_{\text{B}} + \mu_{\text{D}})}{2} \\[4pt]
\text{Parameter form main effect:} \quad & \text{ME}_{\text{param}} = \frac{(\mu_{\text{A}} + \mu_{\text{B}}) - (\mu_{\text{C}} + \mu_{\text{D}})}{2} \\[4pt]
\text{Interaction:} \quad & \text{Int} = (\mu_{\text{A}} - \mu_{\text{B}}) - (\mu_{\text{C}} - \mu_{\text{D}})
\end{aligned}$$

The interaction term $\text{Int}$ is the quantity of central interest. It answers: **does the optimizer effect depend on the parameter form?** If $\text{Int} = 0$, optimizer and parameter form are independent—LoRA works the same regardless of what optimizer you use. If $\text{Int} \neq 0$, the choice of optimizer matters differently under different parameter forms, and the literature's conflation of the two dimensions is a genuine confound.

---

## 4. The Precise Research Question

> **Is LoRA's empirical success attributable to the low-rank constraint itself acting as implicit regularization, or does it depend on being paired with AdamW?**

Formally, we test the null hypothesis:

$$H_0: \text{Int} = 0 \quad \text{(optimizer and parameter form effects are independent)}$$

against the alternative:

$$H_1: \text{Int} \neq 0 \quad \text{(the optimizer effect is moderated by parameter form)}$$

If $H_0$ holds, LoRA's advantage is a structural property of low-rank parameterization—it works with any reasonable optimizer because the rank constraint itself prevents overfitting. The choice of optimizer is secondary.

If $H_1$ holds, the literature's practice of bundling LoRA with AdamW in comparisons is a genuine confound: LoRA might work well with AdamW but poorly with a different optimizer, and attributing performance to LoRA alone would be misleading.

---

## 5. What We Found

### 5.1 The interaction is large and negative

Empirically, $\text{Int} > 10^3$ PPL across all testable architectures. **The null hypothesis is rejected.** Optimizer and parameter form are not independent—but the sign is opposite to what one might expect.

ASP performs *worse* under LoRA than under full-rank, relative to the AdamW baseline. ASP+LoRA (Protocol C) is dominated by AdamW+LoRA (Protocol D) at all step counts. Across 7 independent comparisons, adding ALS to Protocol C consistently degrades performance.

### 5.2 The negative synergy has a structural cause

Under LoRA, ASP's ALS phase solves the full-rank least-squares problem in effective-weight space:

$$W_{\text{new}}^\top = (X^\top X + \lambda I)^{-1} X^\top (X W_{\text{eff}}^\top)$$

at computational cost $\mathcal{O}(N d_{\text{in}}^2 + d_{\text{in}}^3)$—the *same* cost as full-rank ALS. But the solution must then be projected onto the LoRA parameter space via the minimum-norm B-projection:

$$\Delta B = \frac{r}{\alpha} \cdot (W_{\text{new}} - W_{\text{eff}}) \cdot A^\top (A A^\top + \lambda I)^{-1}$$

This projection discards information. The information loss is bounded below by the $(r+1)$-th singular value of $W_{\text{new}} - W_0$:

$$\|W_{\text{new}} - (W_0 + \frac{\alpha}{r}(B + \Delta B)A)\|_F \geq \sigma_{r+1}(W_{\text{new}} - W_0)$$

**ALS pays full-rank computational cost but receives only rank-$r$ information.** The cost-information mismatch is the mathematical root of the negative synergy.

### 5.3 LoRA $r=8$ is sufficient regardless of optimizer

Despite the negative interaction, a deeper pattern emerges: **LoRA $r=8$ reaches a performance plateau under both optimizers**. Increasing rank to $r=256$ or $r=512$ provides zero additional benefit under matched configurations. The rank sufficiency is not optimizer-dependent.

### 5.4 Full-rank fine-tuning overfits regardless of optimizer

Under both ASP and AdamW, full-rank post-training on $N < 10^4$ samples enters the memorization regime. The M-index diagnostic:

$$M = \frac{\text{PPL}_{\text{train}}}{\text{PPL}_{\text{cross}}}, \qquad M < 1 \text{ when } \frac{N_p}{N_d} > 10^4$$

For full-rank, $N_p/N_d \sim 10^5$–$10^6$, so $M < 1$ always. For LoRA $r=8$, $N_p/N_d \sim 10^2$–$10^3$, so $M > 2$ always. The near-perfect WikiText-2 PPL of 1.25 achieved by full-rank is memorization, not generalization—confirmed by downstream degradation on HellaSwag ($-3.2$pp), MMLU ($-4.2$pp), and ARC ($-3.3$pp).

### 5.5 The Rank Sufficiency Law explains why $r=8$ is enough

From the residual stream capacity model, we derive:

$$\boxed{r_{\min} = \eta \cdot \frac{L}{d_h}, \qquad \eta \approx 230}$$

The derivation: each of the $L$ layers experiences a distribution-shift error $\varepsilon(\ell) \propto (L-\ell)/d_h$. Summing over layers gives total correction needed $\propto L^2/(2d_h)$. LoRA provides correction capacity $C_{\text{eff}}(r) = 8r d_h L$ (4 attention modules, input + output dimensions). Equating supply and demand at equilibrium:

$$8 r_{\min} d_h L = \kappa \cdot \frac{L^2}{2d_h} \;\Rightarrow\; r_{\min} = \frac{\kappa}{16} \cdot \frac{L}{d_h}$$

For all currently popular architectures ($L/d_h \leq 0.035$ for strong-pretraining models), this predicts $r_{\min} \leq 8$. The only model exceeding the threshold is SmolLM2-135M ($L/d_h = 0.0521$, predicted $r_{\min} \approx 12$), confirmed by fine-grained calibration.

---

## 6. The Answer

**LoRA works because the low-rank constraint is itself regularization.** The rank $r=8$ bottleneck prevents the model from memorizing the training distribution, forcing it to capture only the dominant directions of the distribution shift—which, for well-pretrained Transformers, are fully captured by $\approx 8$ independent correction directions per layer, as predicted by $r_{\min} = \eta \cdot L/d_h$.

The optimizer is secondary. AdamW is sufficient; a theoretically stronger optimizer (ASP, with exact closed-form solving) cannot overcome the rank bottleneck—it pays full-rank cost for low-rank information, producing negative synergy. The $2 \times 2$ factorial design proves this attribution, and the Rank Sufficiency Law provides the quantitative mechanism.

**The practical rule**: for small-data post-training ($N_d < 10^4$), use LoRA with $r = \max(8, \lceil\eta \cdot L/d_h\rceil)$ and any reasonable optimizer. Never use full-rank—it will overfit. Never trust in-distribution perplexity alone—it measures memorization, not generalization.
