# X2: Causal Depth Boundary Theory — Design Spec

**Date**: 2026-06-23
**Status**: Approved — standalone mathematical note, independent of v3.3 paper
**Output**: `docs/causal_depth_boundary.md` — 4-6 page mathematical derivation

---

## Problem

The existing depth boundary derivation (§A.2, §6.2) is purely phenomenological:
- $\bar{\rho} \approx 1.08$ estimated from fitting digestion times
- $L_{\max} \approx 26$ from "perturbation amplification exceeds SGD recovery"
- No structural explanation for *why* this specific value

## Solution

Frame the residual stream as a **Structural Causal Model (SCM)**. Each layer is a causal mechanism $h_{l+1} = h_l + f_l(h_l)$. ALS performs an intervention on layer $l$'s weights. The intervention effect propagates through $L-l$ downstream layers. The depth boundary is the point where this causal chain's accumulated deviation exceeds the model's self-correction capacity.

## Content Outline

1. **Causal Model of Residual Stream** — Define SCM for transformer layers. Intervention semantics of ALS weight updates. Difference from gradient-based updates (continuous, not interventional).

2. **Intervention Propagation Lemma** — The causal effect of ALS at layer $l$ propagates to layer $L$ as: $\delta_L = \Delta h_l \cdot \prod_{k=l}^{L-1} (I + J_{f_k})$. Bound the norm in terms of per-layer Jacobian spectral radii.

3. **Causal Breakdown Condition** — Define $C_{\text{recovery}} = \eta \cdot T_{\text{SGD}}$ as SGD's self-correction capacity per cycle. Derive $L_{\max}$ from inequality $C_{\text{recovery}} < \|\Delta h_l\| \cdot \prod_{k=l}^{L-1} \rho_k$.

4. **Architectural Predictions** — Derive $L_{\max}$ estimates for: skip-layer models (MoE, $\rho_{\text{eff}} < \rho$), highway connections, encoder-decoder per-stack.

5. **Connection to Causal Representation Learning** — Relate to SCM identifiability (Hyvärinen et al.), causal abstraction, and interventional robustness.

## References to Connect

- Pearl (2009) — SCM foundations
- Noci et al. (2022) — Signal propagation in transformers
- Schölkopf et al. (2021) — Toward causal representation learning
- Existing paper data: 8 architectures, $\bar{\rho} \approx 1.08$, $L_{\max} \approx 26$

## Scope

- Pure derivation — no new experiments
- 4-6 pages mathematical content
- Output: `docs/causal_depth_boundary.md`
- Seed for future standalone paper or workshop submission
