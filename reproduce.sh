#!/bin/bash
# Reproduce key experiments from "Disentangling Optimizer and Parameter Form" (v3.4)
# Requirements: Python 3.12+, 2x RTX 5090 (32GB) for 7B experiments
# Install: pip install -r requirements.txt

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== Reproducibility Pipeline v3.4 ==="
echo "Root: $ROOT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU-only')"

# ── Phase 1: Rank Sufficiency Law (Small Models, ~2h CPU/GPU) ──
run_rank_curve() {
    echo ""
    echo ">>> Phase 1: Rank Curve — Qwen2.5-0.5B (r=8,16,32,64,128,256,512)"
    python experiments/_param_matched_baseline.py
    echo "Output: runs/param_matched_baseline/"
}

# ── Phase 2: Cross-Architecture Validation (~2h GPU) ──
run_cross_arch() {
    echo ""
    echo ">>> Phase 2: Cross-Architecture Rank Curve (5 models)"
    python experiments/_xval.py
    echo "Output: runs/cross_arch/"
}

# ── Phase 3: Falsification Tests (~1h GPU) ──
run_falsify() {
    echo ""
    echo ">>> Phase 3: Falsification Experiments"
    python experiments/_falsify.py
    echo "Output: runs/falsify/"
}

# ── Phase 4: Downstream Evaluation (~3h GPU) ──
run_downstream() {
    echo ""
    echo ">>> Phase 4: Downstream Task Evaluation (HellaSwag, MMLU, ARC)"
    python experiments/_eval_downstream.py
    echo "Output: runs/ (checkpoint-dependent)"
}

# ── Phase 5: C4 Cross-Domain Evaluation (~1h GPU) ──
run_c4() {
    echo ""
    echo ">>> Phase 5: C4 Cross-Domain PPL"
    python experiments/_eval_c4.py
    echo "Output: runs/ (checkpoint-dependent)"
}

# ── Phase 6: η Nomogram Calibration (~20min) ──
run_nomogram() {
    echo ""
    echo ">>> Phase 6: η Nomogram Calibration (GPT-2 + OPT-125m)"
    python experiments/_x3_gpt2_opt.py
    echo "Output: runs/x3_nomogram/"
}

# ── Phase 7: Protocol A Full ASP (~30min CPU, Cholesky ALS) ──
run_full_asp() {
    echo ""
    echo ">>> Phase 7: Full ASP with Cholesky ALS (OPT-125m, 12L)"
    python experiments/_f2_full_asp.py
    echo "Output: runs/f2_full_asp/"
}

# ── Phase 8: Chinese WikiText (~15min) ──
run_chinese() {
    echo ""
    echo ">>> Phase 8: Chinese WikiText-2 Rank Curve"
    python experiments/_p0_chinese_wt.py
    echo "Output: runs/p0_chinese_wt/"
}

# ── Phase 9: Generate Figures ──
run_figures() {
    echo ""
    echo ">>> Phase 9: Generate Figures 1-6"
    python scripts/gen_fig6_nomogram.py
    echo "Output: figures/fig6_nomogram.pdf"
}

# ── Menu ──
case "${1:-all}" in
    all)
        echo "Running ALL experiments (estimated: 8-12h GPU + 2h CPU)"
        run_rank_curve
        run_cross_arch
        run_falsify
        run_downstream
        run_c4
        run_nomogram
        run_full_asp
        run_chinese
        run_figures
        ;;
    rank)
        run_rank_curve ;;
    cross-arch)
        run_cross_arch ;;
    falsify)
        run_falsify ;;
    downstream)
        run_downstream ;;
    c4)
        run_c4 ;;
    nomogram)
        run_nomogram ;;
    full-asp)
        run_full_asp ;;
    chinese)
        run_chinese ;;
    figures)
        run_figures ;;
    quick)
        echo "Quick validation (rank curve + nomogram + figures, ~1h)"
        run_rank_curve
        run_nomogram
        run_figures
        ;;
    *)
        echo "Usage: $0 {all|rank|cross-arch|falsify|downstream|c4|nomogram|full-asp|chinese|figures|quick}"
        exit 1
        ;;
esac

echo ""
echo "=== Pipeline Complete ==="
echo "Results in: runs/"
echo "Figures in: figures/"
echo "Paper: paper/paper_v3.4.tex"
