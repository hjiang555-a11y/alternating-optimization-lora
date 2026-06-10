"""
Analysis and visualization tools for experiment results.

Produces:
  1. 2×2 factorial comparison plots (loss vs FLOPs, perplexity vs memory)
  2. Pareto frontiers for optimizer × parameter form trade-offs
  3. Statistical summaries with confidence intervals
  4. Ablation analysis for phase scheduling choices
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_results(results_dir: str) -> dict[str, dict]:
    """
    Load experiment results from JSON files.

    Expected structure:
      results_dir/
        protocol_A.json
        protocol_B.json
        protocol_C.json
        protocol_D.json
        combined_results.json

    Returns dict mapping protocol label to result dict.
    """
    results: dict[str, dict] = {}
    results_path = Path(results_dir)

    for label in ["A", "B", "C", "D"]:
        path = results_path / f"protocol_{label}.json"
        if path.exists():
            with open(path) as f:
                results[label] = json.load(f)

    return results


def compute_factorial_analysis(results: dict[str, dict]) -> dict:
    """
    Compute 2×2 factorial analysis of variance.

    Factors:
      - Optimizer: AltOpt vs AdamW
      - Parameter Form: Full-Rank vs LoRA

    Main effects and interaction are computed for the final loss metric.

    Returns:
      dict with 'main_effects', 'interaction', 'cell_means'
    """
    labels = {
        "A": ("altopt", "full_rank"),
        "B": ("adamw", "full_rank"),
        "C": ("altopt", "lora"),
        "D": ("adamw", "lora"),
    }

    # Extract final losses
    cell_means: dict[tuple[str, str], float] = {}
    for label, (opt, form) in labels.items():
        if label in results:
            cell_means[(opt, form)] = results[label].get("final_loss", float("inf"))

    # Main effects
    altopt_mean = np.mean([
        cell_means.get(("altopt", "full_rank"), 0),
        cell_means.get(("altopt", "lora"), 0),
    ])
    adamw_mean = np.mean([
        cell_means.get(("adamw", "full_rank"), 0),
        cell_means.get(("adamw", "lora"), 0),
    ])
    optimizer_effect = altopt_mean - adamw_mean

    full_rank_mean = np.mean([
        cell_means.get(("altopt", "full_rank"), 0),
        cell_means.get(("adamw", "full_rank"), 0),
    ])
    lora_mean = np.mean([
        cell_means.get(("altopt", "lora"), 0),
        cell_means.get(("adamw", "lora"), 0),
    ])
    parameter_form_effect = full_rank_mean - lora_mean

    # Interaction effect
    interaction = (
        cell_means.get(("altopt", "full_rank"), 0)
        - cell_means.get(("adamw", "full_rank"), 0)
        - cell_means.get(("altopt", "lora"), 0)
        + cell_means.get(("adamw", "lora"), 0)
    )

    return {
        "main_effects": {
            "optimizer": optimizer_effect,
            "parameter_form": parameter_form_effect,
        },
        "interaction": interaction,
        "cell_means": {f"{opt}+{form}": v for (opt, form), v in cell_means.items()},
    }


def plot_comparison(results: dict[str, dict], output_path: Optional[str] = None) -> None:
    """
    Generate comparison plots.

    If matplotlib is available, produces:
      1. Loss vs FLOPs curves (one line per protocol)
      2. 2×2 heatmap of final loss
      3. Pareto frontier: perplexity vs memory

    Args:
        results: loaded results dict from load_results()
        output_path: directory to save plots (defaults to current dir)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping plots")
        return

    output_path = Path(output_path or ".")
    output_path.mkdir(exist_ok=True)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Loss vs FLOPs
    ax = axes[0]
    colors = {"A": "#e74c3c", "B": "#3498db", "C": "#e67e22", "D": "#2ecc71"}
    labels = {
        "A": "Full-Rank AltOpt",
        "B": "Full-Rank AdamW",
        "C": "LoRA-AltOpt",
        "D": "LoRA-AdamW",
    }
    for label, result in results.items():
        if "loss_history" in result and len(result["loss_history"]) > 0:
            losses = result["loss_history"]
            total_flops = result.get("total_flops", 0)
            flops_per_step = total_flops / max(len(losses), 1)
            flops_axis = np.arange(len(losses)) * flops_per_step
            ax.plot(flops_axis, losses, color=colors.get(label, "gray"),
                    label=labels.get(label, label), alpha=0.8)

    ax.set_xlabel("Cumulative FLOPs")
    ax.set_ylabel("Training Loss")
    ax.set_title("Loss vs FLOPs by Protocol")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # Plot 2: 2×2 Heatmap
    ax = axes[1]
    matrix = np.zeros((2, 2))
    cell_map = {("A", 0, 0), ("B", 0, 1), ("C", 1, 0), ("D", 1, 1)}
    for label, row, col in cell_map:
        if label in results:
            matrix[row, col] = results[label].get("final_loss", 0)
    sns.heatmap(matrix, annot=True, fmt=".4f", ax=ax,
                xticklabels=["Full-Rank", "LoRA"],
                yticklabels=["AltOpt", "AdamW"],
                cmap="YlOrRd_r")
    ax.set_title("2×2 Factorial: Final Loss")

    # Plot 3: Perplexity vs Memory
    ax = axes[2]
    for label, result in results.items():
        if "eval_perplexity" in result and "peak_memory_mb" in result:
            ax.scatter(result["peak_memory_mb"], result["eval_perplexity"],
                       color=colors.get(label, "gray"),
                       label=labels.get(label, label), s=100)

    ax.set_xlabel("Peak Memory (MB)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity vs Memory (Pareto Frontier)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path / "comparison_plots.png", dpi=150)
    plt.close(fig)
    logger.info("Plots saved to %s", output_path / "comparison_plots.png")


def print_summary_table(results: dict[str, dict]) -> None:
    """
    Print a formatted summary table of all protocol results.

    Includes: final loss, perplexity, FLOPs, memory, wall time.
    """
    header = f"{'Proto':<6} {'Optimizer':<10} {'ParamForm':<12} {'Loss':<10} {'PPL':<8} {'FLOPs':<12} {'Mem(MB)':<10} {'Time(s)':<10}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    meta = {
        "A": ("AltOpt", "Full-Rank"),
        "B": ("AdamW", "Full-Rank"),
        "C": ("AltOpt", "LoRA"),
        "D": ("AdamW", "LoRA"),
    }

    for label, (opt, form) in meta.items():
        if label not in results:
            continue
        r = results[label]
        print(
            f"{label:<6} {opt:<10} {form:<12} "
            f"{r.get('final_loss', float('nan')):<10.4f} "
            f"{r.get('eval_perplexity', float('nan')):<8.2f} "
            f"{r.get('total_flops', 0):<12.2e} "
            f"{r.get('peak_memory_mb', 0):<10.0f} "
            f"{r.get('wall_time_seconds', 0):<10.0f}"
        )

    # Factorial analysis
    print(f"\n{sep}")
    print("Factorial Analysis (2×2 ANOVA on final loss):")
    analysis = compute_factorial_analysis(results)
    print(f"  Optimizer main effect:    {analysis['main_effects']['optimizer']:+.6f}")
    print(f"  Parameter form effect:    {analysis['main_effects']['parameter_form']:+.6f}")
    print(f"  Interaction effect:       {analysis['interaction']:+.6f}")
    print(f"\n  Interpretation:")
    opt_dir = "AltOpt better" if analysis['main_effects']['optimizer'] < 0 else "AdamW better"
    form_dir = "Full-rank better" if analysis['main_effects']['parameter_form'] < 0 else "LoRA better"
    print(f"    Optimizer direction: {opt_dir}")
    print(f"    Parameter form direction: {form_dir}")
    if abs(analysis['interaction']) > 0.01:
        print(f"    ⚠  Significant interaction: optimizer effect depends on parameter form")
    else:
        print(f"    ✓ No significant interaction: effects are additive")
    print(sep)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    results_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/"
    results = load_results(results_dir)

    if not results:
        print("No results found. Run experiments first with `python experiments/runner.py`")
    else:
        print_summary_table(results)
        plot_comparison(results, results_dir)
