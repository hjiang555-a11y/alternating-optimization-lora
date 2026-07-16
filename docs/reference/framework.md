# Alternating Optimization Framework: Formal Definition

## 1. Problem Setting

Given a pretrained LLM with parameters $\theta_0 \in \mathbb{R}^d$, and a post-training dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, we seek to find updated parameters $\theta^*$ that minimize the empirical risk:

$$\theta^* = \arg\min_{\theta} \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$$

where $f_\theta$ is the LLM forward function and $\ell$ is the task loss (e.g., cross-entropy for language modeling).

## 2. Alternating Optimization Protocol

The alternating optimization framework defines a **phase schedule** — an ordered sequence of distinct optimization phases, each using a different update mechanism.

### 2.1 Phase Schedule

A phase schedule $\mathcal{S}$ is a sequence:

$$\mathcal{S} = [(P_1, k_1), (P_2, k_2), \ldots, (P_m, k_m)]$$

where $P_j \in \{\text{ALS}, \text{SGD}, \text{Perturb}\}$ and $k_j$ is the number of steps for phase $j$.

The full schedule repeats for $C$ cycles. The total number of parameter updates is:

$$K = C \cdot \sum_{j=1}^m k_j$$

### 2.2 Phase I: Alternating Least Squares (ALS)

**Goal**: Block-wise globally optimal solution (with respect to current activations).

For a linear layer with weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, partition the output dimension into $B = \lceil d_{\text{out}} / b \rceil$ blocks of size $b$:

$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_B \end{bmatrix}, \quad W_j \in \mathbb{R}^{b \times d_{\text{in}}}$$

For each block $j$, fix all other blocks and solve the regularized least squares problem:

$$W_j^* = \arg\min_{W_j} \| X W_j^T - Y_j \|_F^2 + \lambda \| W_j \|_F^2$$

where $X \in \mathbb{R}^{N \times d_{\text{in}}}$ is the input activation matrix (computed via forward pass) and $Y_j$ is the target for block $j$.

**Closed-form solution**:

$$W_j^* = (X^T X + \lambda I)^{-1} X^T Y_j$$

**Computational cost**: $O(N d_{\text{in}}^2 + d_{\text{in}}^3)$ per block (via Cholesky decomposition), or $O(N d_{\text{in}}^2)$ for the Gram matrix $X^T X$ (shared across blocks) plus $O(b d_{\text{in}}^2)$ per-block solve.

### 2.3 Phase II: Stochastic Gradient Descent (SGD)

**Goal**: Fine-grained convergence capturing cross-block interactions.

Standard mini-batch SGD update:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_{\text{mini-batch}}(\theta_t)$$

Optionally with momentum:

$$v_{t+1} = \mu v_t + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

**Computational cost**: $O(d)$ per step (forward + backward pass).

**Key advantage over ALS**: SGD gradients capture cross-block coupling that ALS ignores by solving blocks independently. However, SGD is prone to getting trapped in local minima.

### 2.4 Phase III: Stochastic Perturbation

**Goal**: Escape narrow local minima that trap pure gradient methods.

$$\theta_{t+1} = \theta_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_t^2 I)$$

where $\sigma_t$ follows a decay schedule (e.g., cosine):

$$\sigma_t = \sigma_0 \cdot \frac{1}{2} \left(1 + \cos\left(\frac{\pi t}{T_{\max}}\right)\right)$$

**Computational cost**: $O(d)$ (parameter-space addition, negligible compared to ALS/SGD).

**Layer-wise scaling**: Different layer types receive different perturbation magnitudes:

$$\sigma_t^\ell = \sigma_t \cdot \gamma(\ell), \quad \gamma(\ell) = \begin{cases}
0.1 & \text{embedding layers} \\
0.5 & \text{attention projections} \\
1.0 & \text{feed-forward layers}
\end{cases}$$

The rationale: embedding layers encode discrete semantic information sensitive to perturbation; attention layers have moderate sensitivity; FFN layers have high redundancy.

## 3. Synergy of the Three Phases

The three phases address complementary limitations:

| Limitation | Addressed By | Mechanism |
|------------|-------------|-----------|
| SGD local minima | Perturbation | Parameter-space noise explores surrounding basins |
| ALS ignores cross-block coupling | SGD | Gradients capture global interaction structure |
| SGD slow convergence on ill-conditioned problems | ALS | Closed-form solution for well-conditioned blocks |
| ALS cost explosion | Phase scheduling | ALS runs rarely (1 step per cycle), SGD runs frequently (100 steps) |

The alternating schedule creates a form of **implicit exploration-exploitation trade-off**:

- **ALS**: Exploitation — finds block-optimal solution given current activations
- **SGD**: Exploitation + mild exploration — gradient steps refine the solution
- **Perturbation**: Exploration — escapes current basin for potentially better ones

## 4. Comparison with LoRA

### 4.1 LoRA Parameter Structure

LoRA constrains the parameter update to be low-rank:

$$\Delta W = \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \; A \in \mathbb{R}^{r \times d_{\text{in}}}$$

where $r \ll \min(d_{\text{out}}, d_{\text{in}})$ is the rank hyperparameter.

LoRA is **not an optimizer** — it is a **parameter structure constraint**. By default, LoRA parameters are trained with AdamW.

### 4.2 The Confound

| Variable | AltOpt Controls | LoRA Controls |
|----------|----------------|---------------|
| Parameter form | Full-rank ΔW | Low-rank ΔW = BA |
| Update rule | ALS + SGD + Perturb | AdamW |
| Compute cost | Matrix inversion in ALS phase | Gradient-only |
| Memory | Full-rank gradients | Low-rank adapter states |

Any direct comparison confounds all four variables simultaneously.

### 4.3 Disentanglement via 2×2 Factorial Design

Our proposed solution: cross the two factors (optimizer × parameter form) in a full factorial:

| | Full-Rank ΔW | Low-Rank ΔW |
|---|---|---|
| **AltOpt** | Protocol A | Protocol C |
| **AdamW** | Protocol B | Protocol D |

Comparisons:

- **A vs B**: Is AltOpt a better optimizer than AdamW *given full-rank parameters*?
- **C vs D**: Is AltOpt a better optimizer than AdamW *given low-rank parameters*?
- **A vs C**: Does full-rank parameterization improve AltOpt performance?
- **B vs D**: Does full-rank parameterization improve AdamW performance?
- **Interaction (A-B)-(C-D)**: Does the optimizer effect depend on parameter form?
