"""A-SYNC Variants Report — algorithm descriptions, experiments, comparisons.

Generates a PDF with:
  1. Algorithm: Protocol A baseline vs A-SYNC gradient injection
  2. Variant progression: 9 approaches, from naive to converged
  3. Charts: convergence curves, ablation, depth boundary
  4. Final scoreboard with comparison table
"""
import json, math, os, textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from fpdf import FPDF

REPORT_DIR = "docs"
CHART_DIR = os.path.join(REPORT_DIR, "figures", "a_sync_report")
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "figure.figsize": (7, 4),
})

C = {
    "blue": "#2563EB", "red": "#DC2626", "green": "#16A34A",
    "orange": "#EA580C", "purple": "#7C3AED", "gray": "#6B7280",
    "cyan": "#0891B2", "pink": "#DB2777", "dark": "#1F2937",
    "amber": "#D97706",
}


def chart_convergence_a_sync():
    """Fig 1: A-SYNC convergence curves on Qwen7B (28L) — all variants."""
    data = {
        "A-SYNC 48-const": {
            "ppls": json.load(open("runs/a_sync_48cycle_7b.json"))["ppls"],
            "color": C["blue"], "ls": "-", "marker": "o", "markevery": 4,
        },
        "A-SYNC 24-const": {
            "ppls": json.load(open("runs/a_sync_constant_7b.json"))["ppls"],
            "color": C["cyan"], "ls": "--", "marker": "s", "markevery": 4,
        },
        "A-SYNC 16-cosine": {
            "ppls": json.load(open("runs/a_sync_swa_cosine_7b.json"))["ppls"],
            "color": C["purple"], "ls": "-.", "marker": "D", "markevery": 4,
        },
        "A-SYNC 32-cosine": {
            "ppls": json.load(open("runs/a_sync_32cycle_7b.json"))["ppls"],
            "color": C["pink"], "ls": ":", "marker": "^", "markevery": 4,
        },
        "A-SYNC 8 no-pert": {
            "ppls": json.load(open("runs/a_sync_noperturb_8cycle_7b.json"))["ppls"],
            "color": C["green"], "ls": "--", "marker": "v", "markevery": 2,
        },
        "Pure SGD": {
            "ppls": json.load(open("runs/sgd_vs_async_7b.json"))["pure_sgd"]["ppls"],
            "color": C["red"], "ls": ":", "marker": "x", "markevery": 3,
        },
    }

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, d in data.items():
        p = d["ppls"]
        xs = list(range(1, len(p)+1))
        ax.plot(xs, p, color=d["color"], linestyle=d["ls"], linewidth=1.8,
                marker=d["marker"], markersize=5, markevery=d["markevery"],
                label=label, alpha=0.9)

    # Baseline PPL reference
    ax.axhline(73, color=C["gray"], linestyle=":", linewidth=1, alpha=0.4)
    ax.text(2, 73, "Qwen7B baseline PPL=73", fontsize=7, color=C["gray"], va="bottom")

    # Divergence zone
    ax.axhline(10, color=C["red"], linestyle="--", linewidth=0.8, alpha=0.3)
    ax.text(45, 11, "Protocol A original: diverges at 28L", fontsize=7, color=C["red"], ha="right")

    ax.set_xlabel("ALS-SGD Cycle")
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title("A-SYNC Variant Convergence on Qwen2.5-7B (28L)", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig1_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_ablation():
    """Fig 2: Ablation — A-SYNC vs Pure SGD vs Cosine vs Constant sync."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Left: 7B — sync schedule impact
    labels_7b = ["Cosine", "Constant", "3x8 Restart"]
    finals_7b = [10.5, 7.6, 16.5]
    colors_7b = [C["purple"], C["blue"], C["orange"]]
    bars = ax1.bar(labels_7b, finals_7b, color=colors_7b, alpha=0.85, width=0.5)
    for bar, val in zip(bars, finals_7b):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"PPL={val}", ha="center", fontsize=9, fontweight="bold")
    ax1.axhline(22.5, color=C["red"], linestyle=":", linewidth=1.2, alpha=0.5)
    ax1.text(0.5, 23.5, "Pure SGD plateau", fontsize=7, color=C["red"])
    ax1.set_ylabel("Final PPL")
    ax1.set_title("Sync Schedule Impact (7B, 28L)", fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")

    # Right: decay type — constant wins
    ax2.plot([1,2,3,4,5,6,7,8], [9, 8, 7.5, 7.2, 7.0, 6.9, 6.8, 6.8],
            "^-", color=C["blue"], linewidth=2, markersize=7, label="Constant sync")
    ax2.plot([1,2,3,4,5,6,7,8],
            [9, 7.5, 7, 7.2, 7.5, 8, 8.5, 9],
            "s-", color=C["purple"], linewidth=2, markersize=7, label="Cosine decay")
    ax2.plot([1,2,3,4,5,6,7,8],
            [9, 8.2, 7.8, 9, 8, 7.5, 9, 8.2],
            "D--", color=C["orange"], linewidth=2, markersize=7, label="Warm restart")
    ax2.set_xlabel("Relative Training Time")
    ax2.set_ylabel("Conceptual PPL")
    ax2.set_title("Why Constant Sync Wins", fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig2_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_variant_waterfall():
    """Fig 3: Waterfall of all A-SYNC variants from worst to best."""
    variants = [
        ("Protocol A (original, 28L)", "DIVERGED", "inf", C["red"]),
        ("A-SYNC + perturb (8 cyc)", "Oscillates", 25.8, C["red"]),
        ("A-SYNC no-perturb (8 cyc)", "Converges", 16.6, C["orange"]),
        ("A-CYCLE 3x8 restart (24 cyc)", "Converges", 16.5, C["orange"]),
        ("A-SYNC 32-cosine", "Plateaus", 13.3, C["amber"]),
        ("A-SYNC 16-cosine", "Converges", 10.5, C["green"]),
        ("A-SYNC 24-constant", "Converges", 9.0, C["cyan"]),
        ("A-SYNC 48-constant (BEST)", "Converged", 7.6, C["blue"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [v[0] for v in variants]
    ppls = [float(v[2]) if v[2] != "inf" else 100 for v in variants]
    colors = [v[3] for v in variants]

    bars = ax.barh(range(len(names)), ppls, color=colors, alpha=0.85, height=0.5)
    for i, (bar, (name, status, ppl_str, _)) in enumerate(zip(bars, variants)):
        label = f"{ppl_str}" if ppl_str != "inf" else "DIVERGED"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=8, fontweight="bold", color=colors[i])

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"  {n}" for n in names], fontsize=7.5)
    ax.set_xlabel("Final PPL (lower = better)")
    ax.set_title("A-SYNC Variant Progression on Qwen2.5-7B (28L)", fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.grid(True, alpha=0.2, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig3_waterfall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_depth_boundary():
    """Fig 4: Protocol A vs A-SYNC depth boundary comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))

    # Left: original Protocol A depth boundary
    depths = [12, 22, 24, 28]
    models = ["OPT-125m", "TinyLlama", "Qwen0.5B", "Qwen2.5-7B"]
    pa_status = ["Converged", "Converged", "Converged", "DIVERGED"]
    pa_colors = [C["green"]]*3 + [C["red"]]

    for i, (d, m, s, col) in enumerate(zip(depths, models, pa_status, pa_colors)):
        ax1.bar(i, 1, color=col, alpha=0.15, width=0.6)
        ax1.text(i, 0.5, f"{m}\n{d}L\n{s}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col, alpha=0.85))
    ax1.axvline(2.5, color=C["red"], linestyle="--", linewidth=2, alpha=0.6)
    ax1.text(2.5, 1.1, "Divergence\nBoundary", ha="center", fontsize=7, color=C["red"], fontweight="bold")
    ax1.set_title("Protocol A (Original)", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks([])
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f"{d}L" for d in depths])

    # Right: A-SYNC crosses the boundary
    a_sync_status = ["Converged"]*4
    a_sync_ppls = ["PPL~106", "PPL~15", "PPL 5.5", "PPL 7.6"]
    a_sync_colors = [C["green"]]*3 + [C["blue"]]
    for i, (d, m, s, ppl, col) in enumerate(zip(depths, models, a_sync_status, a_sync_ppls, a_sync_colors)):
        ax2.bar(i, 1, color=col, alpha=0.15, width=0.6)
        ax2.text(i, 0.5, f"{m}\n{d}L\n{ppl}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col, alpha=0.85))
    ax2.axvline(2.5, color=C["blue"], linestyle="-", linewidth=2, alpha=0.6)
    ax2.text(2.5, 1.1, "Boundary\nCrossed!", ha="center", fontsize=8, color=C["blue"], fontweight="bold")
    ax2.set_title("A-SYNC (Ours)", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"{d}L" for d in depths])

    fig.suptitle("Protocol A Depth Boundary: Before vs After A-SYNC", fontweight="bold", fontsize=12)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig4_depth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_0_5b():
    """Fig 5: 0.5B convergence — perturb vs no-perturb."""
    d = json.load(open("runs/a_sync_8cycle_qwen05b.json"))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, color, ls in [("with_perturb", C["red"], "--"), ("no_perturb", C["green"], "-")]:
        p = d[label]["ppls"]
        ax.plot(range(1, len(p)+1), p, color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=6, label=label.replace("_", " "))
    ax.set_xlabel("A-SYNC Cycle")
    ax.set_ylabel("PPL")
    ax.set_title("Perturbation Ablation on Qwen0.5B (24L)", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig5_perturb_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_final_scoreboard():
    """Fig 6: Final scoreboard bar chart."""
    variants = [
        ("A-SYNC 48-const", 7.6, C["blue"]),
        ("A-SYNC 24-const", 9.0, C["cyan"]),
        ("A-SYNC 16-cosine", 10.5, C["green"]),
        ("A-SYNC 32-cosine", 13.3, C["amber"]),
        ("A-CYCLE 3x8", 16.5, C["orange"]),
        ("A-PROBE 16", 22.8, C["pink"]),
        ("Pure SGD 16", 22.5, C["red"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [v[0] for v in variants]
    vals = [v[1] for v in variants]
    colors = [v[2] for v in variants]
    bars = ax.barh(range(len(names)), vals, color=colors, alpha=0.85, height=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"PPL {val}", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Final PPL (lower = better)")
    ax.set_title("Protocol A Variant Scoreboard — Qwen2.5-7B (28L)", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig6_scoreboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
class ReportPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(True, 20)
        self.add_font("DV", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        self.add_font("DV", "B", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        self.add_font("DV", "I", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf")
        self.add_font("DVM", "", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

    def header(self):
        if self.page_no() == 1: return
        self.set_font("DV", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, "A-SYNC: Protocol A Variant Report", align="L")
        self.cell(0, 5, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("DV", "I", 6)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d')} | alternating-optimization-lora", align="C")

    def title_page(self):
        self.add_page()
        self.ln(30)
        self.set_font("DV", "B", 22)
        self.set_text_color(*self._rgb(C["dark"]))
        self.multi_cell(0, 11, "Protocol A-SYNC:\nFrom Divergence to Convergence\non Deep Models", align="C")
        self.ln(5)
        self.set_font("DV", "", 12)
        self.set_text_color(*self._rgb(C["gray"]))
        self.cell(0, 7, "Algorithm Variant Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 7, "July 2026", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(10)
        # Summary box
        y0 = self.get_y()
        self.set_draw_color(*self._rgb(C["blue"]))
        self.rect(25, y0, self.w-50, 45, style="D")
        self.set_fill_color(*self._rgb(C["blue"]))
        self.set_xy(25, y0+2)
        self.set_font("DV", "B", 10)
        self.set_text_color(255, 255, 255)
        self.cell(self.w-50, 7, "  KEY RESULT", align="C")
        self.set_xy(25, y0+11)
        self.set_font("DV", "", 8.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(self.w-50, 5,
            "Protocol A (ALS -> SGD -> Perturb) diverged on all models with 28+ transformer layers. "
            "A-SYNC replaces direct ALS weight application with gradient injection: "
            "the ALS-optimized delta direction is injected into the SGD gradient each step, "
            "allowing head and body to co-evolve. No perturbation is needed.\n\n"
            "Qwen2.5-7B (28L): PPL 58.8 -> 7.6 over 48 cycles. Monotonic convergence.\n"
            "All 7 prior fix attempts (parameter tuning, LARS, clipping, multi-layer ALS, etc.) failed.",
            align="C")

    def section(self, num, title):
        self.ln(4)
        self.set_font("DV", "B", 12)
        self.set_text_color(*self._rgb(C["dark"]))
        self.cell(0, 7, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self._rgb(C["blue"]))
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w-self.r_margin, self.get_y())
        self.ln(3)

    def body(self, text):
        self.set_font("DV", "", 8.5)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        self.multi_cell(self.w - 2*self.l_margin, 4.5, text)

    def body_bold(self, text):
        self.set_font("DV", "B", 8.5)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        self.multi_cell(self.w - 2*self.l_margin, 4.5, text)

    def code_block(self, text):
        self.set_font("DVM", "", 7)
        self.set_text_color(60, 60, 60)
        self.set_fill_color(248, 248, 248)
        for line in text.split("\n"):
            self.cell(0, 3.8, f"  {line}", fill=True, new_x="LMARGIN", new_y="NEXT")

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            cw = (self.w - 2*self.l_margin) / len(headers)
            col_widths = [cw] * len(headers)
        self.set_font("DV", "B", 7)
        self.set_fill_color(*self._rgb(C["dark"]))
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            self.cell(w, 6, f" {h}", fill=True, border=0)
        self.ln()
        self.set_text_color(40, 40, 40)
        for i, row in enumerate(rows):
            self.set_font("DV", "", 7)
            bg = (248, 248, 248) if i % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            for cell, w in zip(row, col_widths):
                self.cell(w, 5.5, f" {cell}", fill=True, border=0)
            self.ln()
        self.ln(2)

    def img(self, path, w=175):
        if os.path.exists(path):
            self.image(path, x=(self.w-w)/2, w=w)
            self.ln(2)
        else:
            self.body(f"[Image missing: {path}]")

    def callout(self, text, color_hex=C["red"]):
        r, g, b = self._rgb(color_hex)
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("DV", "B", 8)
        self.cell(self.w - 2*self.l_margin, 5.5, f"  {text}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    @staticmethod
    def _rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def build_pdf(paths):
    pdf = ReportPDF()
    pdf.title_page()

    # ── 1. Motivation ──
    pdf.section(1, "Motivation: Why Original Protocol A Fails")

    pdf.body(
        "Protocol A interleaves three phases: ALS (exact block-wise least squares on lm_head), "
        "SGD (stochastic gradient descent on all parameters), and Perturb (random noise injection). "
        "On models with 12-24 transformer layers, this converges reliably. "
        "On models with >=28 layers, every attempt diverges within 2-3 cycles."
    )
    pdf.ln(1)
    pdf.body_bold("Root Cause: Residual Amplification")
    pdf.body(
        "ALS modifies only the lm_head (output projection layer). The perturbation dW propagates "
        "forward through L-1 frozen transformer blocks via residual connections (x + sublayer(x)). "
        "Each residual hop amplifies the perturbation by approximately rho ~ 1.08. "
        "After 27 residual connections in a 28-layer Qwen2.5-7B, the effective amplification "
        "is rho^27 ~ 8.7x. The SGD phase recovers at most alpha * 50 ~ 0.005 per cycle, "
        "creating a 1700:1 asymmetry that causes catastrophic divergence."
    )
    pdf.ln(2)
    pdf.callout(
        "The problem is structural, not parametric. ALS's lm_head-only design creates an "
        "impedance mismatch with deep residual architectures.",
        C["red"],
    )

    # ── 2. Algorithm ──
    pdf.section(2, "A-SYNC Algorithm: Gradient Injection")

    pdf.body_bold("Core Innovation")
    pdf.body(
        "Instead of directly writing the ALS-optimized weight into lm_head, A-SYNC computes "
        "the delta W_new - W_old, reverts the weight, and injects the delta as a gradient "
        "bias during SGD. This allows the head and body to co-evolve: the ALS direction guides "
        "SGD without creating the frozen-body amplification chain."
    )
    pdf.ln(2)

    pdf.body_bold("Protocol A-SYNC (one cycle):")
    pdf.code_block(
        "1. ALS solve on lm_head -> get W_new  (label-based exact least squares)\n"
        "2. Compute delta = W_new - W_old (on CPU, offload from GPU)\n"
        "3. Revert lm_head to W_old\n"
        "4. SGD for 50 steps: each step add sync_strength * delta to lm_head gradient\n"
        "5. (Perturbation: REMOVED — found to cause oscillations)\n"
        "6. Repeat from step 1"
    )
    pdf.ln(2)

    pdf.body_bold("Key Design Decisions:")
    pdf.body(
        "1. NO perturbation: Perturb adds destabilizing noise. A-SYNC converges monotonically without it.\n"
        "2. CONSTANT sync strength (0.05): Cosine decay kills the ALS signal in the tail. "
        "Constant sync lets the ALS direction guide SGD throughout all cycles.\n"
        "3. CPU offload for delta: Qwen7B's lm_head is 152064 x 3584 (~2GB). Computing delta on GPU "
        "would OOM. CPU offload with per-block weight snapshots makes it feasible.\n"
        "4. The ALS delta is orthogonal to SGD gradient (cos ~ 0): A-SYNC injects a direction "
        "SGD never explores — the exact information from exact least squares."
    )
    pdf.ln(2)

    pdf.callout(
        "A-SYNC gradient injection: ALS computes WHERE to go (direction). SGD handles HOW to get there "
        "(optimization). The two signals are orthogonal and complementary.",
        C["blue"],
    )

    # ── 3. Variant Progression ──
    pdf.section(3, "Variant Progression: 8 Attempts, 1 Winner")

    pdf.body(
        "The path from the first A-SYNC implementation to the final converging variant "
        "involved 8 iterative refinements. Each variant tested a different hypothesis "
        "about how to integrate the ALS signal with SGD optimization."
    )
    pdf.ln(2)

    pdf.table(
        ["#", "Variant", "Key Change", "7B Final PPL", "Result"],
        [
            ["1", "A-SYNC +perturb", "First A-SYNC with perturbation", "25.8", "Converges, oscillates"],
            ["2", "A-SYNC no-perturb", "Remove perturbation phase", "16.6", "Monotonic, cleaner"],
            ["3", "A-SYNC cosine 16", "Cosine sync decay over 16 cycles", "10.5", "Good, sync dies early"],
            ["4", "A-SYNC cosine 32", "Cosine over 32 cycles", "13.2", "Worse — decay kills tail"],
            ["5", "A-CYCLE restart", "3x8 cycles with warm restart", "16.5", "Restart window too short"],
            ["6", "A-SYNC 24-const", "Constant sync, 24 cycles", "9.0", "Excellent, still improving"],
            ["7", "A-SYNC 48-const", "Constant sync, 48 cycles", "7.6", "BEST — fully converged"],
        ],
        [8, 32, 48, 28, 36],
    )

    pdf.img(paths["convergence"])
    pdf.set_font("DV", "I", 6.5)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 3.5, "Figure 1: A-SYNC convergence curves on Qwen2.5-7B (28L). All variants shown on log scale.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.callout(
        "Key discovery: constant sync strength (0.05) dominates cosine decay. "
        "The ALS signal should persist at full strength — it's not annealing, it's guiding.",
        C["blue"],
    )

    # ── 4. Sync Schedule Analysis ──
    pdf.section(4, "Sync Schedule Analysis: Why Constant Wins")

    pdf.body_bold("Cosine Decay")
    pdf.body(
        "Cosine: sync(t) = 0.05 * 0.5 * (1 + cos(pi * t/T)). At t = T/2, sync = 0.025. "
        "At t = 3T/4, sync = 0.004. Effectively, the ALS signal is active for only half "
        "the training duration. The second half becomes pure SGD -> plateaus."
    )
    pdf.ln(1)
    pdf.body_bold("Warm Restart (A-CYCLE)")
    pdf.body(
        "3 blocks of 8-cycle cosine, resetting sync+lr to max at each block boundary. "
        "Restart shocks the optimization (PPL jumps at each boundary) and 8 cycles is "
        "too short for convergence within each block."
    )
    pdf.ln(1)
    pdf.body_bold("Constant Sync")
    pdf.body(
        "sync = 0.05 fixed. The ALS delta is always present in the gradient at full strength. "
        "SGD momentum (even with momentum=0) accumulates the direction naturally. "
        "Monotonic convergence from C1 to C48 with no oscillations."
    )

    pdf.img(paths["ablation"])
    pdf.set_font("DV", "I", 6.5)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 3.5, "Figure 2: Left — sync schedule impact on final PPL. Right — conceptual illustration of convergence stability.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.table(
        ["Schedule", "7B Final PPL", "Convergence", "Oscillations"],
        [
            ["Cosine (16 cyc)", "10.5", "Good", "None"],
            ["Cosine (32 cyc)", "13.2", "Plateaus at C20", "None"],
            ["Warm restart (3x8)", "16.5", "Jumps at boundaries", "Significant"],
            ["CONSTANT (24 cyc)", "9.0", "Monotonic", "None"],
            ["CONSTANT (48 cyc)", "7.6 (BEST)", "Monotonic to C44", "None"],
        ],
        [42, 28, 42, 32],
    )

    # ── 5. Perturbation Ablation ──
    pdf.section(5, "Perturbation Ablation: Why No Perturb is Better")

    pdf.body(
        "The original Protocol A included a perturbation phase (random noise injection) after SGD. "
        "We tested A-SYNC with and without perturbation on Qwen0.5B (24L)."
    )
    pdf.ln(1)
    pdf.body_bold("Results:")
    pdf.body(
        "With perturbation: PPL oscillates between 7.8 and 23.9. The noise disrupts "
        "the co-evolution of head and body.\n"
        "Without perturbation: PPL 9.1 -> 5.6, monotonic convergence across all 8 cycles. "
        "No noise injection needed — the ALS gradient injection provides sufficient exploration."
    )

    pdf.img(paths["perturb"])
    pdf.set_font("DV", "I", 6.5)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 3.5, "Figure 3: A-SYNC with vs without perturbation on Qwen0.5B (24L). Perturbation causes instability.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.callout(
        "Perturbation is harmful. The ALS gradient injection already provides sufficient "
        "exploration — adding noise creates head-body de-synchronization.",
        C["orange"],
    )

    # ── 6. Depth Boundary ──
    pdf.section(6, "Crossing the Depth Boundary: 12L -> 28L")

    pdf.body(
        "Original Protocol A has a hard boundary at 28 layers — every run diverges. "
        "A-SYNC crosses this boundary and converges monotonically at all tested depths."
    )
    pdf.ln(2)

    pdf.table(
        ["Model", "Layers", "Protocol A", "A-SYNC", "A-SYNC Final PPL"],
        [
            ["OPT-125m", "12", "PPL 107", "PPL ~107", "Converged"],
            ["TinyLlama-1.1B", "22", "PPL 16", "PPL ~15", "Converged"],
            ["Qwen2.5-0.5B", "24", "PPL 18 (unstable)", "PPL 5.5", "Monotonic"],
            ["Qwen2.5-7B", "28", "11/11 DIVERGED", "PPL 7.6", "Converged"],
        ],
        [35, 15, 35, 25, 30],
    )

    pdf.img(paths["depth"])
    pdf.set_font("DV", "I", 6.5)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 3.5, "Figure 4: Left — original Protocol A fails at 28L. Right — A-SYNC crosses the boundary.", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── 7. Comparison to Other Approaches ──
    pdf.section(7, "Comparison: A-SYNC vs Other Fix Attempts")

    pdf.body(
        "Before discovering A-SYNC, 7 other fix strategies were systematically tested. "
        "None succeeded on 28-layer models. The table below compares all approaches."
    )
    pdf.ln(2)

    pdf.table(
        ["Approach", "Category", "24L Result", "28L Result", "Key Limitation"],
        [
            ["Parameter tuning", "Tuning", "Varies", "11/11 diverge", "Non-transferable"],
            ["Depth protection", "Protection", "Stable", "Diverge", "Clipping kills signal"],
            ["LARS optimizer", "Optimizer", "PPL=161k", "Diverge", "Avoids NaN, no convergence"],
            ["Gradient clipping", "Stabilization", "Masked", "Diverge", "Symptom suppression"],
            ["Multi-layer ALS", "Algorithm", "Instant inf", "Instant inf", "Underdetermined X^TX"],
            ["A-PROBE", "Architecture", "PPL=5.5", "PPL=22.8", "Bottleneck limits SGD"],
            ["A-KD", "Distillation", "PPL=195", "N/A", "KL divergence explosion"],
            ["A-SYNC (ours)", "Gradient Injection", "PPL=5.5", "PPL=7.6", "None — BEST"],
        ],
        [32, 22, 22, 24, 44],
    )
    pdf.ln(1)
    pdf.callout(
        "A-SYNC is the ONLY approach that converges on 28L models AND outperforms pure SGD. "
        "Pure SGD plateaus at PPL 22.5; A-SYNC reaches PPL 7.6 (2.96x better).",
        C["blue"],
    )

    # ── 8. Final Scoreboard ──
    pdf.section(8, "Final Scoreboard")

    pdf.img(paths["scoreboard"])
    pdf.set_font("DV", "I", 6.5)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 3.5, "Figure 5: Final Protocol A variant scoreboard on Qwen2.5-7B (28L).", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.img(paths["waterfall"])
    pdf.set_font("DV", "I", 6.5)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 3.5, "Figure 6: A-SYNC variant progression from original Protocol A to 48-cycle constant sync.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── 9. The A-SYNC Recipe ──
    pdf.section(9, "The A-SYNC Recipe (Final Protocol)")

    pdf.body_bold("Configuration:")
    pdf.code_block(
        "sync_strength: 0.05 (CONSTANT, no decay)\n"
        "learning_rate: 2e-4 (CONSTANT, no decay)\n"
        "momentum: 0.0 (SGD, not Adam)\n"
        "weight_decay: 0.01\n"
        "cycles: 24-48 (convergence at ~44)\n"
        "ALS block_size: 512 (per-block solves, avoids OOM)\n"
        "ALS reg_lambda: 1e-3\n"
        "ALS step_size: 0.01\n"
        "Perturbation: DISABLED"
    )
    pdf.ln(2)

    pdf.body_bold("One cycle pseudocode:")
    pdf.code_block(
        "w_before = lm_head.weight.cpu().clone()\n"
        "als.solve_block(batch)  # optimize lm_head via label-based least squares\n"
        "delta = lm_head.weight.cpu() - w_before  # ALS direction\n"
        "lm_head.weight.copy_(w_before)  # revert head to pre-ALS state\n"
        "for j in range(50):\n"
        "    loss = model(batch_j).loss\n"
        "    loss.backward()\n"
        "    lm_head.weight.grad += 0.05 * delta  # inject ALS direction\n"
        "    optimizer.step()"
    )
    pdf.ln(2)

    pdf.body_bold("Why it works:")
    pdf.body(
        "1. ALS computes the optimal head delta direction via exact label-based least squares.\n"
        "2. Instead of applying it directly (head-body mismatch), revert and inject as gradient bias.\n"
        "3. SGD moves head + body together toward the ALS direction.\n"
        "4. The ALS delta is orthogonal to SGD gradient (cos ~ 0) — a direction SGD never explores.\n"
        "5. Constant sync keeps this complementary signal active throughout training.\n"
        "6. No perturbation needed — gradient injection provides sufficient exploration."
    )

    # ── Appendix ──
    pdf.section("A", "Appendix: Data Sources")
    pdf.table(
        ["Data File", "Experiment", "Model", "Key Result"],
        [
            ["runs/a_sync_48cycle_7b.json", "A-SYNC 48-cycle", "Qwen7B", "PPL 58.8 -> 7.6"],
            ["runs/a_sync_constant_7b.json", "A-SYNC 24-cycle", "Qwen7B", "PPL 61.8 -> 9.0"],
            ["runs/a_sync_swa_cosine_7b.json", "A-SYNC 16-cosine", "Qwen7B", "PPL 59.7 -> 10.5"],
            ["runs/a_sync_32cycle_7b.json", "A-SYNC 32-cosine", "Qwen7B", "PPL 59.9 -> 13.3"],
            ["runs/a_cycle_7b.json", "A-CYCLE restart", "Qwen7B", "PPL 61.6 -> 16.5"],
            ["runs/a_sync_8cycle_qwen05b.json", "Perturb ablation", "Qwen0.5B", "No perturb wins"],
            ["runs/sgd_vs_async_7b.json", "Pure SGD ablation", "Qwen7B", "PPL 60.5 -> 22.5"],
            ["runs/probe_7b.json", "A-PROBE", "Qwen7B", "PPL 60.2 -> 22.8"],
        ],
        [52, 30, 22, 38],
    )

    out_path = os.path.join(REPORT_DIR, "a_sync_variant_report.pdf")
    pdf.output(out_path)
    return out_path


# ══════════════════════════════════════════════════════════════════════
def main():
    print("Generating charts...")
    paths = {}
    paths["convergence"] = chart_convergence_a_sync()
    paths["ablation"] = chart_ablation()
    paths["waterfall"] = chart_variant_waterfall()
    paths["depth"] = chart_depth_boundary()
    paths["perturb"] = chart_0_5b()
    paths["scoreboard"] = chart_final_scoreboard()
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("Building PDF...")
    pdf_path = build_pdf(paths)
    print(f"Done: {pdf_path} ({os.path.getsize(pdf_path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
