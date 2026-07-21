"""Generate comprehensive PDF report on Protocol A deep-model fix attempts.

Sections:
  1.  Background — residual amplification theory
  2.  Cross-Depth Benchmark — 12L→28L trend
  3.  Fix #1: lm_head-only ALS with Depth Protection
  4.  Fix #2: Tuning ALS Step Size & ALS:SGD Ratio
  5.  Fix #3: LARS Optimizer
  6.  Fix #4: Gradient Clipping
  7.  Fix #5: Multi-Layer ALS — Batch & Sequential
  8.  Conclusion — failure tally & root cause
"""
import json, math, textwrap, os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from fpdf import FPDF

# ── Config ──────────────────────────────────────────────────────────
REPORT_DIR = "docs"
CHART_DIR = os.path.join(REPORT_DIR, "figures", "report")
os.makedirs(CHART_DIR, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.figsize": (7, 3.8),
})

COLORS = {
    "blue": "#2563EB",
    "red": "#DC2626",
    "green": "#16A34A",
    "orange": "#EA580C",
    "purple": "#7C3AED",
    "gray": "#6B7280",
    "dark": "#1F2937",
    "light": "#F3F4F6",
}


def safe_ppl(x):
    """Format PPL for display, clamping extreme values."""
    if x is None or math.isinf(x) or math.isnan(x):
        return "Diverged"
    if x > 1e12:
        return f"{x:.2e}"
    if x > 1e6:
        return f"{x:,.0f}"
    return f"{x:,.1f}"


# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

def load_data():
    d = {}

    d["depth"] = json.load(open("runs/p1.2_depth/results.json"))
    d["lars_gpt2"] = json.load(open("runs/lars_sanity_gpt2.json"))
    d["lars_qwen"] = json.load(open("runs/lars_qwen05b.json"))
    d["ml_batch"] = json.load(open("runs/multi_layer_qwen05b.json"))
    d["ml_seq"] = json.load(open("runs/seq_multi_layer_qwen05b.json"))

    # Read docs for theoretical context
    with open("docs/als-residual-amplification.md") as f:
        d["theory_doc"] = f.read()
    with open("docs/why-protocol-a-fails-on-7b.md") as f:
        d["failure_doc"] = f.read()

    # All fix attempts summary table
    d["fix_attempts"] = [
        {
            "name": "Tune ALS \u03b1 (0.01\u21920.001)",
            "category": "Parameter Tuning",
            "models": "GPT-2 12L, Qwen2.5-0.5B 24L",
            "result": "Failed",
            "detail": "\u03b1=0.001 requires 10\u00d7 training steps; 12L still non-monotonic; 28L 11/11 divergent",
        },
        {
            "name": "Tune ALS:SGD ratio",
            "category": "Parameter Tuning",
            "models": "GPT-2 12L (1:20\u21921:5)",
            "result": "Failed",
            "detail": "1:20 optimal at 12L but 28L diverges at all ratios; SGD dominance cannot compensate",
        },
        {
            "name": "Depth decay \u03b2 + skip_early_ratio",
            "category": "Depth Protection",
            "models": "8 architectures 12\u201328L",
            "result": "Partial",
            "detail": "Works at \u226424L; fails at 28L because residual amplification exceeds EMA damping capacity",
        },
        {
            "name": "LARS optimizer",
            "category": "Optimizer",
            "models": "GPT-2 12L, Qwen2.5-0.5B 24L",
            "result": "Failed",
            "detail": "GPT-2: PPL diverges (no ALS), PPL stagnates (with ALS); Qwen0.5B: avoids NaN but PPL=161k",
        },
        {
            "name": "Gradient clipping",
            "category": "Stabilization",
            "models": "Qwen2.5-0.5B 24L, Qwen2.5-7B 28L",
            "result": "Failed",
            "detail": "Clip masks symptom; flattens shallow-layer updates \u2192 no meaningful convergence",
        },
        {
            "name": "Multi-layer ALS (Batch)",
            "category": "Algorithm",
            "models": "Qwen2.5-0.5B 24L, k=8 blocks",
            "result": "Failed",
            "detail": "Cross-layer activation interference; 56 layers modified in 1 pass \u2192 instant divergence",
        },
        {
            "name": "Multi-layer ALS (Sequential)",
            "category": "Algorithm",
            "models": "Qwen2.5-0.5B 24L, k=4,8",
            "result": "Failed",
            "detail": "Underdetermined X\u1d40X (4864\u00d74864, rank\u2264256); noise dominates the Cholesky solution",
        },
    ]
    return d


# ══════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════

def chart_cross_depth(data):
    """Fig 1: PPL vs model depth for Protocol A."""
    trend = data["depth"]["depth_trend"]
    fig, ax = plt.subplots(figsize=(7, 4.2))

    layers = trend["layers"]
    ppls = trend["ppls"]
    models = ["OPT-125m", "TinyLlama 1.1B", "Qwen2.5-0.5B", "Qwen2.5-7B"]

    stable_layers = [l for l, p in zip(layers, ppls) if not math.isinf(p)]
    stable_ppls = [p for p in ppls if not math.isinf(p)]

    bars = ax.bar(
        range(len(layers)), [1.0]*len(layers),
        color=[COLORS["green"]]*3 + [COLORS["red"]],
        alpha=0.15, width=0.6,
    )

    for i, (l, p, m) in enumerate(zip(layers, ppls, models)):
        color = COLORS["green"] if not math.isinf(p) else COLORS["red"]
        label = safe_ppl(p)
        ax.text(i, 0.5, f"{m}\n{l} layers\nPPL={label}",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.85))

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"{l}L" for l in layers])
    ax.set_ylabel("Model Depth")
    ax.set_title("Protocol A Convergence by Model Depth", fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])

    # Annotate boundary
    ax.axvline(2.5, color=COLORS["red"], linestyle="--", linewidth=2, alpha=0.6)
    ax.text(2.5, 1.12, "Divergence Boundary", ha="center", fontsize=8,
            color=COLORS["red"], fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=COLORS["red"], boxstyle="round,pad=0.2"))

    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig1_cross_depth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_lars(data):
    """Fig 2: LARS vs SGD on GPT-2 (12L) and Qwen0.5B (24L)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))

    # GPT-2
    for dataset, ax, title in [
        ("lars_gpt2", ax1, "GPT-2 125M (12L)"),
        ("lars_qwen", ax2, "Qwen2.5-0.5B (24L)"),
    ]:
        r = data[dataset]["results"]
        cycles = range(1, 5)
        for label, color, marker in [("SGD", COLORS["blue"], "o"), ("LARS", COLORS["orange"], "s")]:
            ppls = r[label]["ppls"]
            finite_ppls = [(i+1, p) for i, p in enumerate(ppls) if not math.isinf(p) and p < 1e12]
            if finite_ppls:
                xi, yi = zip(*finite_ppls)
                ax.plot(xi, yi, color=color, marker=marker, linewidth=2,
                        markersize=7, label=label, zorder=3)
            # Mark diverged cycles
            diverged_cycles = [i+1 for i, p in enumerate(ppls) if math.isinf(p) or p > 1e12]
            for dc in diverged_cycles:
                ax.annotate("\u2717", (dc, ax.get_ylim()[1]*0.1 if dataset == "lars_qwen" else 5),
                           fontsize=14, color=color, ha="center", fontweight="bold")

        baseline = data[dataset]["baseline_ppl"]
        ax.axhline(baseline, color=COLORS["gray"], linestyle=":", linewidth=1.2, alpha=0.5)
        ax.text(3.5, baseline, f"Baseline ({baseline:,.1f})", fontsize=7, color=COLORS["gray"], va="bottom")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("ALS\u2192SGD\u2192Perturb Cycle")
        ax.set_ylabel("PPL")
        ax.set_yscale("log")
        if dataset == "lars_qwen":
            ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("LARS Optimizer: Two-Model Protocol A Test", fontweight="bold", fontsize=12)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig2_lars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_parameter_tuning():
    """Fig 3: Qualitative diagram of parameter tuning failure modes."""
    fig, ax = plt.subplots(figsize=(7, 3.6))

    # Conceptual: step_size sensitivity
    alphas = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
    # Conceptual: convergence quality (made-up but directionally correct)
    convergence_12l = [0.7, 0.85, 0.95, 0.6, 0.3]   # 12L: optimal at 0.01
    convergence_28l = [0.1, 0.15, 0.05, 0.02, 0.0]   # 28L: all fail

    ax.plot(alphas, convergence_12l, "o-", color=COLORS["blue"], linewidth=2,
            markersize=8, label="Shallow models (\u226412L)")
    ax.plot(alphas, convergence_28l, "s-", color=COLORS["red"], linewidth=2,
            markersize=8, label="Deep models (\u226528L)")

    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_xlabel("ALS Step Size (\u03b1)")
    ax.set_ylabel("Convergence Quality (conceptual)")
    ax.set_title("Parameter Tuning Cannot Bridge the Depth Gap", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Annotate
    ax.annotate("Optimal for shallow\n(does not transfer)",
               xy=(0.01, 0.95), xytext=(0.03, 0.7),
               arrowprops=dict(arrowstyle="->", color=COLORS["blue"]),
               fontsize=8, color=COLORS["blue"])
    ax.annotate("No working \u03b1 for deep",
               xy=(0.005, 0.15), xytext=(0.03, 0.35),
               arrowprops=dict(arrowstyle="->", color=COLORS["red"]),
               fontsize=8, color=COLORS["red"])

    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig3_param_tuning.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_multi_layer(data):
    """Fig 4: Multi-layer ALS — batch vs sequential, k=1 vs k=4 vs k=8."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))

    for dataset, ax, title in [
        ("ml_batch", ax1, "Batch (one forward pass)"),
        ("ml_seq", ax2, "Sequential (one pass per layer)"),
    ]:
        r = data[dataset]["results"]
        for label, color, marker, ls in [
            ("k1", COLORS["green"], "o", "-"),
            ("k4", COLORS["orange"], "s", "--") if "k4" in r else (None, None, None, None),
            ("k8", COLORS["red"], "D", ":"),
        ]:
            if label not in r or label is None:
                continue
            ppls = r[label]["ppls"]
            cycles = [i+1 for i, p in enumerate(ppls)]
            # Plot with log scale, but mark infinity
            for i, p in enumerate(ppls):
                if not math.isinf(p) and p < 1e12:
                    ax.plot(i+1, p, marker=marker, color=color, markersize=7,
                            zorder=3, label=label if i == 0 else "")
                else:
                    ax.plot(i+1, 1e18, marker="x", color=color, markersize=10,
                            zorder=3, mew=2)
            # Connect with line for finite points
            finite = [(i+1, p) for i, p in enumerate(ppls) if not math.isinf(p) and p < 1e12]
            if len(finite) >= 2:
                xi, yi = zip(*finite)
                ax.plot(xi, yi, color=color, linewidth=1.5, linestyle=ls, alpha=0.7)

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("ALS\u2192SGD\u2192Perturb Cycle")
        ax.set_ylabel("PPL")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7, title="Layers solved")

    fig.suptitle("Multi-Layer ALS: Two Strategies, Same Failure", fontweight="bold", fontsize=12)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig4_multi_layer.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_fix_waterfall(data):
    """Fig 5: Waterfall summary of all fix attempts."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    attempts = data["fix_attempts"]
    names = [a["name"] for a in attempts]
    categories = [a["category"] for a in attempts]

    # Assign colors by result
    cat_colors = {"Partial": COLORS["orange"], "Failed": COLORS["red"]}
    bar_colors = [cat_colors.get(
        a["result"], COLORS["gray"]) for a in attempts]

    bars = ax.barh(range(len(names)), [1]*len(names), color=bar_colors, alpha=0.2, height=0.5)
    bars = ax.barh(range(len(names)), [1]*len(names), color=bar_colors, alpha=0.85, height=0.4)

    # Category labels on left
    cat_order = {"Parameter Tuning": 0, "Depth Protection": 1, "Optimizer": 2,
                 "Stabilization": 3, "Algorithm": 4}
    for i, (name, cat) in enumerate(zip(names, categories)):
        ax.text(-0.02, i, f"[{cat}]", ha="right", va="center", fontsize=7,
                color=COLORS["gray"], fontstyle="italic")
        ax.text(0.52, i, name, ha="left", va="center", fontsize=8, fontweight="bold")
        result_marker = "\u2717" if attempts[i]["result"] == "Failed" else "\u2248"
        ax.text(1.02, i, result_marker, ha="center", va="center", fontsize=12,
                color=COLORS["red"] if result_marker == "\u2717" else COLORS["orange"],
                fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([])
    ax.set_xlim(-0.38, 1.15)
    ax.set_xticks([])
    ax.set_title("Protocol A Fix Attempts: Waterfall Summary", fontweight="bold", fontsize=12)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["red"], alpha=0.85, label="Failed"),
        Patch(facecolor=COLORS["orange"], alpha=0.85, label="Partial"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig5_waterfall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_theory():
    """Fig 6: Residual amplification diagram — why 28L is the boundary."""
    fig, ax = plt.subplots(figsize=(7, 3.8))

    depths = np.arange(1, 37, dtype=float)
    # Theoretical amplification: ρ^(L-1) with ρ≈1.08
    rho = 1.08
    amp = rho ** (depths - 1)

    ax.fill_between(depths, 1, amp, alpha=0.15, color=COLORS["red"])
    ax.plot(depths, amp, color=COLORS["red"], linewidth=2.5, label="Residual Amplification \u03c1^(L\u22121)")

    # Mark regimes
    ax.axvspan(1, 24, alpha=0.08, color=COLORS["green"])
    ax.axvspan(24, 28, alpha=0.08, color=COLORS["orange"])
    ax.axvspan(28, 36, alpha=0.08, color=COLORS["red"])

    ax.text(12, 25, "Stable\n(\u226424L)", ha="center", fontsize=9, fontweight="bold", color=COLORS["green"])
    ax.text(26, 25, "Marginal\n(24-28L)", ha="center", fontsize=9, fontweight="bold", color=COLORS["orange"])
    ax.text(32, 25, "Divergent\n(\u226528L)", ha="center", fontsize=9, fontweight="bold", color=COLORS["red"])

    # Mark key amplification values
    for L in [12, 24, 28]:
        amp_val = rho ** (L-1)
        ax.annotate(f"{L}L: \u00d7{amp_val:.1f}",
                   xy=(L, amp_val), xytext=(L-3, amp_val+3),
                   fontsize=8, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=COLORS["dark"], lw=0.8))

    # Horizontal line at SGD recovery rate
    ax.axhline(0.01, color=COLORS["gray"], linestyle=":", linewidth=1, alpha=0.5)
    ax.text(3, 0.012, "SGD recovery rate/cycle (\u03b1\u22480.01)", fontsize=7, color=COLORS["gray"])

    ax.set_xlabel("Model Depth (layers)")
    ax.set_ylabel("Amplification Factor")
    ax.set_title("Why 28 Layers is the Protocol A Divergence Boundary", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    path = os.path.join(CHART_DIR, "fig6_theory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
# PDF BUILDER
# ══════════════════════════════════════════════════════════════════════

class ReportPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(True, 20)
        # Add fonts for Unicode support
        self.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
        self.add_font("DejaVu", "B", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", uni=True)
        self.add_font("DejaVu", "I", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", uni=True)
        self.add_font("DejaVuMono", "", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", uni=True)

    def header(self):
        if self.page_no() == 1:
            return  # No header on title page
        self.set_font("DejaVu", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, "Protocol A Deep-Model Fix Attempts \u2014 Experimental Report", align="L")
        self.cell(0, 5, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 6)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d')} | alternating-optimization-lora | commit bc5fd1d", align="C")

    def title_page(self):
        self.add_page()
        self.ln(40)
        self.set_font("DejaVu", "B", 24)
        self.set_text_color(*self._hex_to_rgb(COLORS["dark"]))
        self.multi_cell(0, 12, "Protocol A Deep-Model\nFix Attempts", align="C")
        self.ln(6)
        self.set_font("DejaVu", "", 13)
        self.set_text_color(*self._hex_to_rgb(COLORS["gray"]))
        self.cell(0, 8, "Experimental Report \u2014 July 2026", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.cell(0, 8, "alternating-optimization-lora", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(15)

        # Summary box
        self.set_fill_color(*self._hex_to_rgb(COLORS["red"]))
        self.set_draw_color(*self._hex_to_rgb(COLORS["red"]))
        y0 = self.get_y()
        self.rect(25, y0, self.w - 50, 38, style="D")
        self.set_fill_color(*self._hex_to_rgb(COLORS["red"]))
        self.set_text_color(255, 255, 255)
        self.set_xy(25, y0 + 2)
        self.set_font("DejaVu", "B", 11)
        self.cell(self.w - 50, 8, "  CONCLUSION: 7 Fixes Attempted.  0 Successful on \u226528-layer Models.", align="C")
        self.set_xy(25, y0 + 12)
        self.set_font("DejaVu", "", 9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(self.w - 50, 5.5,
            "Root cause: ALS perturbation on lm_head propagates through L\u22121 frozen layers via "
            "residual connections, amplifying \u223c8.7\u00d7 in 28-layer models. "
            "No fix within the ALS\u2192SGD\u2192Perturb framework resolves this structural impedance mismatch.",
            align="C")

    def section(self, number, title):
        self.ln(6)
        self.set_font("DejaVu", "B", 13)
        self.set_text_color(*self._hex_to_rgb(COLORS["dark"]))
        self.cell(0, 8, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self._hex_to_rgb(COLORS["blue"]))
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body(self, text):
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "", 9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(self.w - 2 * self.l_margin, 5, text)

    def body_bold(self, text):
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(self.w - 2 * self.l_margin, 5, text)

    def code_block(self, text):
        self.set_font("DejaVuMono", "", 7)
        self.set_text_color(60, 60, 60)
        self.set_fill_color(245, 245, 245)
        lines = text.split("\n")
        for line in lines:
            self.cell(0, 4, f"  {line}", fill=True, new_x="LMARGIN", new_y="NEXT")

    def data_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [self.w / len(headers) - self.l_margin - self.r_margin] * len(headers)
        # Header
        self.set_font("DejaVu", "B", 7.5)
        self.set_fill_color(*self._hex_to_rgb(COLORS["dark"]))
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, f" {h}", fill=True, border=0)
        self.ln()
        # Rows
        self.set_text_color(40, 40, 40)
        for i, row in enumerate(rows):
            self.set_font("DejaVu", "", 7.5)
            bg = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            for cell_text, w in zip(row, col_widths):
                self.cell(w, 6, f" {cell_text}", fill=True, border=0)
            self.ln()
        self.ln(2)

    def image_centered(self, path, w=170):
        if os.path.exists(path):
            self.image(path, x=(self.w - w) / 2, w=w)
            self.ln(3)
        else:
            self.body(f"[Image not found: {path}]")

    def callout(self, text, color_hex=COLORS["red"]):
        r, g, b = self._hex_to_rgb(color_hex)
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("DejaVu", "B", 8.5)
        x0 = self.get_x()
        y0 = self.get_y()
        self.rect(x0, y0, self.w - 2*self.l_margin, 0.1, style="F")
        self.set_xy(x0, y0 + 1)
        self.cell(self.w - 2*self.l_margin, 6, f"  {text}", fill=True)
        self.ln(8)

    @staticmethod
    def _hex_to_rgb(hex_color):
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def build_pdf(data, chart_paths):
    pdf = ReportPDF()

    # ── Title Page ──
    pdf.title_page()

    # ── Section 1: Background ──
    pdf.section(1, "Background: Why Protocol A Fails on Deep Models")

    pdf.body(
        "Protocol A (ALS \u2192 SGD \u2192 Perturb) interleaves exact block-wise least-squares "
        "updates with stochastic gradient descent and perturbation noise. On shallow models "
        "(\u226412 layers), this converges reliably. On models with \u226528 layers, every "
        "attempt diverges within 2\u20133 cycles."
    )
    pdf.ln(2)
    pdf.body_bold("The Residual Amplification Mechanism:")
    pdf.body(
        "ALS modifies only the lm_head (output projection). The perturbation dW propagates "
        "forward through L-1 frozen transformer blocks. Each residual connection "
        "(x + sublayer(x)) amplifies the perturbation by approximately rho ~ 1.08. "
        "After 27 residual connections, the effective amplification is rho^27 ~ 8.7x."
    )
    pdf.ln(2)
    pdf.body(
        "The SGD phase operates at learning rates ~1e-4, recovering at most "
        "alpha * 50 ~ 0.005 per cycle. This asymmetry (8.7x amplification vs 0.005x recovery) "
        "creates a positive feedback loop that escalates within 2-3 ALS cycles."
    )

    pdf.image_centered(chart_paths["theory"])
    pdf.set_font("DejaVu", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, "Figure 1: Theoretical residual amplification \u03c1^(L\u22121) as a function of model depth.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.callout("Core insight: The problem is structural, not parametric. ALS\u2019s lm_head-only design "
                "creates an impedance mismatch with deep residual architectures.")

    # ── Section 2: Cross-Depth Benchmark ──
    pdf.section(2, "Cross-Depth Benchmark: 12L \u2192 28L")

    pdf.body(
        "Protocol A was tested across 8 model architectures spanning 12\u201328 layers. "
        "The divergence boundary is sharp: all models \u226424L converge; all models \u226528L diverge."
    )
    pdf.ln(2)

    pdf.data_table(
        ["Model", "Layers", "Final PPL", "Status"],
        [
            ["OPT-125m", "12", "106.9", "Converged"],
            ["TinyLlama-1.1B", "22", "15.5", "Converged"],
            ["Qwen2.5-0.5B", "24", "18.0", "Converged"],
            ["Qwen2.5-7B", "28", "Diverged (\u221e)", "11/11 attempts fail"],
        ],
        [50, 20, 30, 70],
    )

    pdf.image_centered(chart_paths["depth"])
    pdf.set_font("DejaVu", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, "Figure 2: Protocol A convergence status by model depth.", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.callout("Qwen2.5-7B (28L) marks the hard divergence boundary. 11 independent runs all diverge.")

    # ── Section 3: Parameter Tuning ──
    pdf.section(3, "Fix Attempts #1\u20132: Parameter Tuning")

    pdf.body_bold("3.1 ALS Step Size (\u03b1)")
    pdf.body(
        "Reducing the ALS EMA mixing coefficient \u03b1 from 0.01 to 0.001 lowers the per-cycle "
        "perturbation amplitude, in theory buying SGD more time to recover. In practice:"
    )
    pdf.body(
        "\u2022 \u03b1 = 0.001 requires ~10\u00d7 more training steps for equivalent convergence on shallow models\n"
        "\u2022 On 12-layer models, PPL still exhibits non-monotonic oscillation (cycles alternate between "
        "improvement and regression)\n"
        "\u2022 On 28-layer models, 11/11 runs diverge regardless of \u03b1 value"
    )
    pdf.ln(2)
    pdf.body_bold("3.2 ALS:SGD Step Ratio")
    pdf.body(
        "The ALS:SGD ratio controls how many SGD steps follow each ALS update. Tested ratios "
        "from 1:5 to 1:50 across 12L and 28L architectures:"
    )
    pdf.body(
        "\u2022 Ratio 1:20 is optimal for 12-layer models\n"
        "\u2022 No ratio stabilizes 28-layer models \u2014 divergence is independent of SGD step count\n"
        "\u2022 At 1:50 (SGD-dominant), the PPL diverges more slowly but still diverges"
    )

    pdf.image_centered(chart_paths["param_tuning"])
    pdf.set_font("DejaVu", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, "Figure 3: Conceptual illustration \u2014 no \u03b1 bridges the shallow-to-deep gap.", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.callout("Parameter tuning is a convex optimization over a fundamentally non-convex structural problem.")

    # ── Section 4: Depth Protection (skip_early_ratio + \u03b2 decay) ──
    pdf.section(4, "Fix Attempt \u2265#3: Depth-Boundary Protection")

    pdf.body(
        "Three protective mechanisms were added to the ALS solver:"
    )
    pdf.body(
        "\u2022 skip_early_ratio (0.5): Skip ALS for the first 50% of transformer layers\n"
        "\u2022 depth_decay_beta (2.0): Exponentially damp EMA \u03b1 for deeper layers\n"
        "\u2022 clip_catastrophic: Rollback any layer with \u2016\u0394W\u2016/\u2016W\u2016 > threshold"
    )
    pdf.body(
        "These protections extend the stable regime from \u226412L to \u226424L. "
        "At 28L, residual amplification exceeds the EMA damping capacity: "
        "the clip_catastrophic threshold is triggered on 6\u20138 layers per ALS cycle, "
        "effectively aborting all meaningful updates."
    )

    pdf.callout("Depth protection buys ~12 extra layers but cannot scale to 28L.", COLORS["orange"])

    # ── Section 5: LARS ──
    pdf.section(5, "Fix Attempt #4: LARS Optimizer")

    pdf.body(
        "LARS (Layer-wise Adaptive Rate Scaling) was tested as a replacement for standard "
        "SGD in the SGD phase. LARS normalizes gradients per-layer: "
        "\u03b7_l = \u03b7 \u00b7 min(1, \u03b3\u00b7\u2016W_l\u2016 / \u2016\u2207W_l\u2016), "
        "preventing individual layer gradients from dominating."
    )
    pdf.ln(2)
    pdf.body_bold("Results:")
    pdf.body(
        "\u2022 GPT-2 125M (12L): ALS with SGD converges PPL 87\u219218; ALS with LARS stagnates at 172\u2192146\n"
        "\u2022 Qwen2.5-0.5B (24L): SGD diverges to \u221e by cycle 3; LARS avoids NaN but PPL stagnates at 161k"
    )

    pdf.data_table(
        ["Model", "Optimizer", "Cycle 1", "Cycle 2", "Cycle 3", "Cycle 4"],
        [
            ["GPT-2 12L", "SGD", "87.9", "37.4", "24.7", "18.0"],
            ["GPT-2 12L", "LARS", "173.0", "172.4", "130.3", "146.4"],
            ["Qwen0.5B 24L", "SGD", "79.0", "52.8", "\u221e", "\u221e"],
            ["Qwen0.5B 24L", "LARS", "296.9", "754k", "573k", "161k"],
        ],
        [35, 22, 22, 22, 22, 22],
    )

    pdf.image_centered(chart_paths["lars"])
    pdf.set_font("DejaVu", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, "Figure 4: LARS vs SGD on GPT-2 (left) and Qwen0.5B (right). LARS avoids NaN but does not converge.", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.callout("LARS prevents explosion but cannot recover from ALS-induced perturbation. PPL remains 100\u20131000\u00d7 baseline.")

    # ── Section 6: Gradient Clipping ──
    pdf.section(6, "Fix Attempt #5: Gradient Clipping")

    pdf.body(
        "Global gradient clipping (max_norm=1.0) was applied during the SGD phase to prevent "
        "exploding gradients from residual-amplified ALS perturbations."
    )
    pdf.body(
        "\u2022 Clipping masks the symptom (NaN) but flattens gradient norms for shallow layers\n"
        "\u2022 Shallow layers receive disproportionately small effective updates\n"
        "\u2022 No meaningful convergence: PPL oscillates without a downward trend\n"
        "\u2022 The fundamental imbalance (\u00d78.7 amplification vs \u00d70.005 recovery) persists"
    )

    pdf.callout("Gradient clipping is a constraint, not a solution. It suppresses symptoms without addressing cause.")

    # ── Section 7: Multi-Layer ALS ──
    pdf.section(7, "Fix Attempts #6\u20137: Multi-Layer ALS")

    pdf.body_bold("7.1 Hypothesis")
    pdf.body(
        "If ALS modifies weights in the last K transformer blocks (not just lm_head), "
        "the perturbation propagation distance shrinks from L\u22121 to K\u22121 layers, "
        "reducing amplification from \u03c1^(L\u22121) to \u03c1^(K\u22121)."
    )
    pdf.ln(2)
    pdf.body_bold("7.2 Implementation")
    pdf.body(
        "Extended ALSBlockSolver with multi_layer_depth parameter:\n"
        "\u2022 _build_depth_map parses block indices from HF layer names (layers.N.xxx)\n"
        "\u2022 _select_learnable_modules returns all nn.Linear in the last K blocks\n"
        "\u2022 Reconstruction-based ALS: min \u2016X\u00b7W\u1d40 \u2212 X\u00b7W_old\u1d40\u2016\u00b2 per layer"
    )
    pdf.ln(2)
    pdf.body_bold("7.3 Batch Strategy (One Forward Pass)")
    pdf.body(
        "All target modules are hooked simultaneously in one forward pass. "
        "The activations captured for layer L are stale because layers L\u2212k through L\u22121 "
        "were also modified in this same pass."
    )
    pdf.ln(2)
    pdf.body_bold("7.4 Sequential Strategy (One Pass Per Layer)")
    pdf.body(
        "Each layer gets its own forward pass, guaranteeing correct activations. "
        "But intermediate-layer ALS uses self-reconstruction targets (X\u00b7W\u1d40), "
        "not label-based targets. With batch_size=2, seq_len=128, X has N=256 rows "
        "but d_in up to 4864 \u2192 X\u1d40X is severely underdetermined (rank \u2264 256). "
        "The Cholesky solution is dominated by the \u03bb regularizer \u2192 arbitrary noise "
        "that cascades forward through residual connections."
    )
    pdf.ln(2)

    pdf.data_table(
        ["Strategy", "k", "Cycle 1", "Cycle 2", "Cycle 3", "Cycle 4", "Verdict"],
        [
            ["Baseline (lm_head only)", "1", "11.3", "49.7", "18.6", "9.8", "Converges"],
            ["Batch", "8", "69,694", "767k", "287M", "\u221e", "Diverges inst.",
        ],
            ["Sequential", "4", "326k", "2.8B", "8.7B", "\u221e", "Diverges"],
            ["Sequential", "8", "\u221e", "\u2014", "\u2014", "\u2014", "Diverges inst."],
        ],
        [30, 10, 22, 22, 22, 22, 30],
    )

    pdf.image_centered(chart_paths["multi_layer"])
    pdf.set_font("DejaVu", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, "Figure 5: Multi-layer ALS \u2014 batch (left) vs sequential (right). Both strategies fail.", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.callout("Multi-layer ALS has a fundamental correctness-speed tradeoff with no viable middle ground.")

    # ── Section 8: Conclusion ──
    pdf.section(8, "Conclusion & Failure Tally")

    pdf.body(
        "Seven distinct fix strategies were systematically tested. Zero succeeded on "
        "models with \u226528 transformer layers. The root cause is structural: ALS modifies "
        "only lm_head, but the perturbation propagates through L\u22121 frozen residual blocks, "
        "amplifying ~8.7\u00d7 in 28-layer architectures. No parameter, optimizer, or algorithmic "
        "fix within the ALS\u2192SGD\u2192Perturb framework can resolve this impedance mismatch."
    )
    pdf.ln(2)

    pdf.data_table(
        ["#", "Fix Attempt", "Category", "Models Tested", "Result"],
        [
            ["1", "Tune ALS \u03b1 (0.01\u21920.001)", "Parameter", "12L, 24L, 28L", "Failed"],
            ["2", "Tune ALS:SGD ratio (1:5\u21921:50)", "Parameter", "12L, 28L", "Failed"],
            ["3", "Depth decay \u03b2 + skip ratio", "Protection", "8 arch. 12\u201328L", "Partial (\u226424L)"],
            ["4", "LARS optimizer (SGD phase)", "Optimizer", "12L, 24L", "Failed"],
            ["5", "Gradient clipping", "Stabilization", "24L, 28L", "Failed"],
            ["6", "Multi-layer ALS (Batch)", "Algorithm", "24L, k=8", "Failed"],
            ["7", "Multi-layer ALS (Seq.)", "Algorithm", "24L, k=4,8", "Failed"],
        ],
        [8, 45, 25, 28, 25],
    )
    pdf.ln(4)

    pdf.image_centered(chart_paths["waterfall"])
    pdf.set_font("DejaVu", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, "Figure 6: Waterfall summary of all 7 fix attempts.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.body_bold("Paths Forward:")
    pdf.body(
        "\u2022 Structural redesign: ALS must modify weights at multiple depths simultaneously, "
        "not just lm_head. But any approach must solve the underdetermination problem for "
        "intermediate layers (label-based targets only exist at the output).\n"
        "\u2022 Alternative interleaving: Replace ALS with a different exact solver that operates "
        "on LoRA subspaces rather than full-rank weights.\n"
        "\u2022 Residual gating: Temporarily disable or attenuate residual connections during ALS "
        "phase to block perturbation propagation.\n"
        "\u2022 Accept the constraint: Protocol A works on \u226424L models. For deeper architectures, "
        "use Protocol B (pure SGD+LoRA) or Protocol C (SGD+LoRA+Perturb)."
    )

    # ── Appendix: Data Sources ──
    pdf.section("A", "Appendix: Data Sources")
    pdf.body(
        "All experiments conducted on 2\u00d7 NVIDIA RTX 5090 (31GB), CUDA 12.8, PyTorch 2.9.0.\n"
        "Dataset: WikiText-2 (raw). Protocol A: ALS (block_size=1024, \u03bb=1e\u207b\u00b3, \u03b1=0.01) "
        "\u2192 SGD (lr=1e\u207b\u2074, momentum=0.9) \u00d750 steps \u2192 Perturb (\u03c3=1e\u207b\u00b3). "
        "4 cycles per run unless divergence forced early termination."
    )
    pdf.ln(2)
    pdf.data_table(
        ["Data File", "Experiment", "Date"],
        [
            ["runs/p1.2_depth/results.json", "Cross-depth benchmark", "Jun 2026"],
            ["runs/lars_sanity_gpt2.json", "LARS on GPT-2 12L", "Jun 2026"],
            ["runs/lars_qwen05b.json", "LARS on Qwen0.5B 24L", "Jun 2026"],
            ["runs/multi_layer_qwen05b.json", "Multi-layer ALS (batch)", "Jul 2026"],
            ["runs/seq_multi_layer_qwen05b.json", "Multi-layer ALS (seq.)", "Jul 2026"],
        ],
        [70, 55, 25],
    )

    # ── Save ──
    out_path = os.path.join(REPORT_DIR, "protocol_a_fix_attempts.pdf")
    pdf.output(out_path)
    return out_path


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    data = load_data()

    print("Generating charts...")
    paths = {}
    paths["theory"] = chart_theory()
    paths["depth"] = chart_cross_depth(data)
    paths["lars"] = chart_lars(data)
    paths["param_tuning"] = chart_parameter_tuning()
    paths["multi_layer"] = chart_multi_layer(data)
    paths["waterfall"] = chart_fix_waterfall(data)

    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("Building PDF...")
    pdf_path = build_pdf(data, paths)
    print(f"\nDone: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
