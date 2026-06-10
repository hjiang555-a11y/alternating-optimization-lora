# Related Work

## Alternating Optimization

- **Alternating Least Squares (ALS)** has a long history in matrix factorization for collaborative filtering (Koren et al., 2009). The key insight — partitioning parameters into blocks and solving each exactly while holding others fixed — generalizes naturally to neural network weight matrices partitioned by output rows.

- **Block Coordinate Descent (BCD)** (Tseng, 2001; Nesterov, 2012) provides theoretical convergence guarantees for alternating block-wise optimization. Under certain conditions (convexity, block separability), BCD converges to the global optimum. For neural networks (non-convex), BCD converges to a stationary point under Lipschitz gradient assumptions.

- **Alternating Direction Method of Multipliers (ADMM)** (Boyd et al., 2011) is another alternating framework that decomposes problems into sub-problems. ADMM has been applied to neural network training (Taylor et al., 2016) but primarily for convex-regularized objectives.

## Low-Rank Adaptation

- **LoRA** (Hu et al., 2021) introduced the low-rank decomposition $\Delta W = BA$ for fine-tuning large language models. The key innovation was demonstrating that the "intrinsic dimension" of task-specific adaptation is much lower than the full parameter count.

- **Variants**:
  - **AdaLoRA** (Zhang et al., 2023): Adaptive rank allocation across layers
  - **QLoRA** (Dettmers et al., 2023): 4-bit quantization + LoRA for memory efficiency
  - **DoRA** (Liu et al., 2024): Weight-decomposed low-rank adaptation separating magnitude and direction
  - **LoRA+** (Hayou et al., 2024): Asymmetric learning rates for A and B matrices

- **Theoretical analyses**:
  - Aghajanyan et al. (2020) showed that the intrinsic dimension of fine-tuning is ~0.01-0.1× the full parameter count
  - Malladi et al. (2023) analyzed the loss landscape of LoRA-adapted models, finding that LoRA projects gradients onto a smooth manifold

## Perturbation-Based Optimization

- **Noisy Gradient Descent** adds Gaussian noise to gradients: $\theta_{t+1} = \theta_t - \eta(\nabla L + \xi_t)$. This has connections to Langevin dynamics and can help escape sharp minima (Zhu et al., 2019).

- **Sharpness-Aware Minimization (SAM)** (Foret et al., 2021) perturbs parameters to find flatter minima: $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t + \rho \nabla L(\theta_t) / \|\nabla L(\theta_t)\|)$

- **Stochastic Weight Averaging (SWA)** (Izmailov et al., 2018) averages checkpoints along the optimization trajectory for better generalization.

- **Warm Restarts / Cyclical LR** (Loshchilov & Hutter, 2017) periodically increase learning rate to "jump" basins — analogous to the perturbation phase but via learning rate rather than direct noise.

## Fair Comparison in ML Systems

- **NAS-Bench** (Ying et al., 2019) and **MLPerf** (Mattson et al., 2020) established protocols for fair comparison in neural architecture search and training systems.

- **Tango** (Groeneveld et al., 2023) and **OLMO** (Groeneveld et al., 2024) advocate for fully reproducible training pipelines with fixed random seeds, data ordering, and environment.

- Our 2×2 factorial approach is inspired by **factorial experimental design** from statistics (Fisher, 1935), which disentangles main effects from interactions when multiple factors are varied simultaneously.

## Open Questions

1. Can ALS block solves be approximated efficiently (e.g., via randomized SVD) to reduce the cost gap with gradient-based methods?

2. Does the low-rank manifold amplify or dampen the benefits of stochastic perturbation?

3. Is there a "Pareto-optimal" schedule of ALS:SGD:Perturb ratios that generalizes across model scales?

4. Can the alternating framework be combined with LoRA (Protocol C) to get the best of both worlds — low-rank efficiency + alternating optimization benefits?

5. How does the choice of matrix inversion method (Cholesky vs iterative CG vs randomized sketching) affect the FLOPs-accuracy trade-off in the ALS phase?
