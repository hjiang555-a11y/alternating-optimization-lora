"""
Visualization toolkit for AltOpt vs LoRA experiments.

Produces publication-quality figures:
  1. Training curves with phase-annotated regions
  2. FLOPs-Perplexity Pareto frontiers
  3. 2x2 factorial heatmaps
  4. ALS:SGD ratio ablation bar charts
  5. Perturbation effect waterfall plots
  6. Generalization gap comparison

All plots use matplotlib + seaborn with consistent styling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


PROTOCOL_COLORS = {
    "A": "#e74c3c",
    "B": "#3498db",
    "C": "#e67e22",
    "D": "#2ecc71",
}

PROTOCOL_LABELS = {
    "A": "Full-Rank + AltOpt",
    "B": "Full-Rank + AdamW",
    "C": "LoRA + AltOpt",
    "D": "LoRA + AdamW",
}


def _get_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError:
        return None, None


def plot_training_curves(results: dict, output_path: str, title: str = "Training Loss Curves"):
    plt, sns = _get_matplotlib()
    if plt is None:
        logger.warning("matplotlib not available")
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, data in results.items():
        losses = data.get("loss_history", [])
        if not losses:
            continue
        ax.plot(range(len(losses)), losses, color=PROTOCOL_COLORS.get(label, "gray"),
                label=PROTOCOL_LABELS.get(label, label), linewidth=2, alpha=0.85)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, frameon=True)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", output_path)


def plot_pareto_frontier(data: list[dict], output_path: str):
    """
    FLOPs vs Perplexity Pareto frontier.

    Args:
        data: list[dict] with keys 'label', 'flops', 'perplexity', 'peak_memory_mb'
    """
    plt, sns = _get_matplotlib()
    if plt is None:
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: FLOPs vs Perplexity
    ax = axes[0]
    for d in data:
        label = d.get("label", "?")
        ax.scatter(d.get("flops", 0), d.get("perplexity", 0),
                   color=PROTOCOL_COLORS.get(label, "gray"),
                   label=PROTOCOL_LABELS.get(label, label),
                   s=150, edgecolors="black", linewidth=1, zorder=3)
    ax.set_xlabel("Total FLOPs", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("FLOPs vs Perplexity (Pareto Frontier)", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xscale("log")

    # Plot 2: Memory vs Perplexity
    ax = axes[1]
    for d in data:
        label = d.get("label", "?")
        ax.scatter(d.get("peak_memory_mb", 0), d.get("perplexity", 0),
                   color=PROTOCOL_COLORS.get(label, "gray"),
                   label=PROTOCOL_LABELS.get(label, label),
                   s=150, edgecolors="black", linewidth=1, zorder=3)
    ax.set_xlabel("Peak GPU Memory (MB)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("Memory vs Perplexity", fontsize=14)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pareto frontier saved to %s", output_path)


def plot_factorial_heatmap(cell_values: dict, output_path: str, metric: str = "Loss"):
    """
    2x2 factorial design heatmap.

    Args:
        cell_values: {(optimizer, parameter_form): float}
    """
    plt, sns = _get_matplotlib()
    if plt is None:
        return

    matrix = np.zeros((2, 2))
    for (opt, form), val in cell_values.items():
        row = 0 if opt == "altopt" else 1
        col = 0 if form == "full_rank" else 1
        matrix[row, col] = val

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", ax=ax,
                xticklabels=["Full-Rank", "LoRA"],
                yticklabels=["AltOpt", "AdamW"],
                cmap="YlOrRd_r", linewidths=2, linecolor="white",
                cbar_kws={"label": metric})
    ax.set_title(f"2x2 Factorial: {metric}", fontsize=14, pad=15)
    ax.set_xlabel("Parameter Form", fontsize=12)
    ax.set_ylabel("Optimizer", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Factorial heatmap saved to %s", output_path)


def plot_ratio_ablation(ratio_results: dict, output_path: str):
    """
    ALS:SGD ratio ablation bar chart.

    Args:
        ratio_results: {ratio_key: {final_perplexity: float, total_flops: float, ...}}
    """
    plt, sns = _get_matplotlib()
    if plt is None:
        return

    ratios = list(ratio_results.keys())
    ppls = [ratio_results[r]["final_perplexity"] for r in ratios]
    flops = [ratio_results[r].get("total_flops", 0) for r in ratios]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("viridis", len(ratios))
    bars = ax1.bar(ratios, ppls, color=colors, edgecolor="black", linewidth=1)
    ax1.set_xlabel("ALS:SGD Ratio", fontsize=12)
    ax1.set_ylabel("Perplexity", fontsize=12, color="#2c3e50")
    ax1.tick_params(axis="y", labelcolor="#2c3e50")

    ax2 = ax1.twinx()
    ax2.plot(ratios, flops, "D-", color="#e74c3c", linewidth=2, markersize=10,
             label="Total FLOPs")
    ax2.set_ylabel("Total FLOPs", fontsize=12, color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    for bar, ppl in zip(bars, ppls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{ppl:.1f}", ha="center", fontsize=10, fontweight="bold")

    ax1.set_title("ALS:SGD Ratio Ablation — Perplexity vs FLOPs", fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Ratio ablation plot saved to %s", output_path)


def plot_generalization_gap(protocol_data: dict, output_path: str):
    """
    Generalization gap bar chart (eval_loss - train_loss per protocol).

    Args:
        protocol_data: {label: {final_train_loss, final_eval_loss, generalization_gap}}
    """
    plt, sns = _get_matplotlib()
    if plt is None:
        return

    labels_order = ["A", "B", "C", "D"]
    labels = [l for l in labels_order if l in protocol_data]
    train_losses = [protocol_data[l]["final_train_loss"] for l in labels]
    eval_losses = [protocol_data[l]["final_eval_loss"] for l in labels]
    gaps = [protocol_data[l]["generalization_gap"] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, train_losses, width, label="Train Loss",
                   color="#3498db", edgecolor="black", linewidth=1, alpha=0.9)
    bars2 = ax.bar(x + width / 2, eval_losses, width, label="Eval Loss",
                   color="#e74c3c", edgecolor="black", linewidth=1, alpha=0.9)

    for i, gap in enumerate(gaps):
        ax.annotate(f"gap={gap:.3f}", (x[i], max(train_losses[i], eval_losses[i]) + 0.1),
                    ha="center", fontsize=9, fontweight="bold",
                    color="#e67e22" if gap > 0.5 else "#27ae60")

    ax.set_xlabel("Protocol", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Generalization Gap: Train vs Eval Loss", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS.get(l, l) for l in labels], rotation=20, ha="right")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Generalization gap plot saved to %s", output_path)


def plot_perturbation_effect(perturb_events: list[dict], loss_history: list[float],
                              output_path: str):
    """
    Loss curve with perturbation event markers.

    Args:
        perturb_events: list of {step, loss_before, noise_energy}
        loss_history: full loss values per step
    """
    plt, sns = _get_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    steps = range(len(loss_history))
    ax.plot(steps, loss_history, color="#2c3e50", linewidth=2, label="Training Loss")

    for event in perturb_events:
        step = event["step"]
        loss_before = event["loss_before"]
        ax.axvline(x=step, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate("Perturb", (step, loss_before),
                    xytext=(step + 2, loss_before * 1.1),
                    fontsize=8, color="#e74c3c",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8))

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Perturbation Effect on Loss Trajectory", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Perturbation effect plot saved to %s", output_path)


def generate_all_plots(ablation_results: dict, output_dir: str = "figures/"):
    """
    Generate all visualization plots from ablation results.

    Args:
        ablation_results: output of experiments/ablation.py run_all_ablation()
        output_dir: directory to save PNG files
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # RQ2: Efficiency frontier
    if "RQ2" in ablation_results and "error" not in ablation_results["RQ2"]:
        r2 = ablation_results["RQ2"]
        plot_pareto_frontier([
            {"label": "A", "flops": r2["protocol_a"]["total_flops"],
             "perplexity": r2["protocol_a"]["final_perplexity"],
             "peak_memory_mb": r2["protocol_a"]["peak_memory_mb"]},
            {"label": "B", "flops": r2["protocol_b"]["total_flops"],
             "perplexity": r2["protocol_b"]["final_perplexity"],
             "peak_memory_mb": r2["protocol_b"]["peak_memory_mb"]},
        ], str(output / "rq2_pareto.png"))

    # RQ3: Perturbation effect
    if "RQ3" in ablation_results and "error" not in ablation_results["RQ3"]:
        r3 = ablation_results["RQ3"]
        events = r3.get("perturbation_events", [])
        losses = r3["with_perturbation"].get("loss_history", [])
        if events and losses:
            plot_perturbation_effect(events, losses, str(output / "rq3_perturbation.png"))

    # RQ4: Generalization gap
    if "RQ4" in ablation_results and "error" not in ablation_results["RQ4"]:
        r4 = ablation_results["RQ4"]
        plot_generalization_gap(r4["protocols"], str(output / "rq4_generalization.png"))

    # RQ6: ALS:SGD ratio
    if "RQ6" in ablation_results and "error" not in ablation_results["RQ6"]:
        r6 = ablation_results["RQ6"]
        plot_ratio_ablation(r6["ratios"], str(output / "rq6_ratio_ablation.png"))

    logger.info("All plots generated in %s", output_dir)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    results_path = sys.argv[1] if len(sys.argv) > 1 else "runs/ablation/ablation_results.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "figures/"

    with open(results_path) as f:
        results = json.load(f)

    generate_all_plots(results, output_dir)
    print(f"Plots saved to {output_dir}")
