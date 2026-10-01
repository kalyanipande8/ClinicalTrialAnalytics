"""
Leadership-facing visualization dashboard.

Produces a single, annotated static figure (output/figures/leadership_dashboard.png)
summarizing, for a non-technical clinical-leadership audience:
  1. Overall treatment effect (mixed-effects estimate, with CI, vs. the naive
     pooled estimate) -- "does the drug work, and how confident are we."
  2. Per-site treatment effect estimates (forest-plot style) -- "is the effect
     consistent across sites, or driven by one or two sites."
  3. Site enrollment vs. data quality -- "which sites can we trust, which need
     a monitoring visit."
  4. Data-quality metrics by site with the problem site clearly flagged.

Colorblind-safe categorical palette, no rainbow or dual-axis charts, direct
annotations so the figure is readable without a legend-decoding exercise.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Colorblind-safe palette (Okabe-Ito derived): blue = primary/good, orange = warning,
# vermillion = critical/problem, gray = neutral/reference
COLOR_GOOD = "#0072B2"      # blue
COLOR_WATCH = "#E69F00"     # orange
COLOR_PROBLEM = "#D55E00"   # vermillion
COLOR_NEUTRAL = "#888888"   # gray
COLOR_TREATMENT = "#009E73" # green


def build_dashboard():
    mixed = pd.read_csv("output/mixed_model_results.csv").iloc[0]
    site_effects = pd.read_csv("output/per_site_treatment_effects.csv")
    dqa = pd.read_csv("output/dqa_site_summary.csv")

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        "Multi-Site Clinical Trial: Treatment Effect & Data Quality Summary",
        fontsize=17, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.955,
        "Synthetic demonstration data | Primary endpoint = change from baseline (lower = better) | "
        "Prepared for clinical leadership review",
        ha="center", fontsize=9.5, color="#555555", style="italic",
    )

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.38, wspace=0.28,
                           top=0.91, bottom=0.07, left=0.08, right=0.96)

    # --- Panel 1: Overall treatment effect, mixed model vs naive ---
    ax1 = fig.add_subplot(gs[0, 0])
    estimates = ["Naive pooled\n(ignores site)", "Mixed-effects model\n(accounts for site)"]
    values = [mixed["naive_ols_treatment_effect"], mixed["mixed_model_treatment_effect"]]
    errs = [1.96 * mixed["naive_ols_se"], 1.96 * (mixed["mixed_model_ci_upper"] - mixed["mixed_model_treatment_effect"]) / 1.0]
    ci_low = [values[0] - errs[0], mixed["mixed_model_ci_lower"]]
    ci_high = [values[0] + errs[0], mixed["mixed_model_ci_upper"]]
    colors = [COLOR_NEUTRAL, COLOR_TREATMENT]
    y_pos = [0, 1]
    for i, (v, lo, hi, c) in enumerate(zip(values, ci_low, ci_high, colors)):
        ax1.plot([lo, hi], [i, i], color=c, linewidth=3, solid_capstyle="round")
        ax1.plot(v, i, "o", color=c, markersize=11, zorder=3)
        ax1.text(v, i + 0.22, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color=c)
    ax1.axvline(0, color="#999999", linewidth=1, linestyle="--")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(estimates, fontsize=10)
    ax1.set_xlabel("Treatment effect (Treatment - Control)\nNegative = favors Treatment", fontsize=9)
    ax1.set_title("Overall Treatment Effect Estimate", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.7, 1.7)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # --- Panel 2: Per-site forest plot ---
    ax2 = fig.add_subplot(gs[0, 1])
    se_df = site_effects.sort_values("treatment_effect")
    ys = np.arange(len(se_df))
    for i, row in enumerate(se_df.itertuples()):
        lo, hi = row.treatment_effect - 1.96 * row.se, row.treatment_effect + 1.96 * row.se
        ax2.plot([lo, hi], [i, i], color=COLOR_GOOD, linewidth=2, alpha=0.8)
        ax2.plot(row.treatment_effect, i, "o", color=COLOR_GOOD, markersize=6)
    ax2.axvline(mixed["mixed_model_treatment_effect"], color=COLOR_TREATMENT, linewidth=2,
                linestyle="--", label="Overall (mixed model)")
    ax2.axvline(0, color="#999999", linewidth=1, linestyle=":")
    ax2.set_yticks(ys)
    ax2.set_yticklabels(se_df["site_id"], fontsize=8.5)
    ax2.set_xlabel("Per-site treatment effect estimate (95% CI)", fontsize=9)
    ax2.set_title("Treatment Effect Consistency Across Sites", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    # --- Panel 3: Enrollment vs data quality (bubble) ---
    ax3 = fig.add_subplot(gs[1, 0])
    status_color = {"OK": COLOR_GOOD, "WATCH": COLOR_WATCH, "PROBLEM SITE - review required": COLOR_PROBLEM}
    for _, row in dqa.iterrows():
        c = status_color[row["dqa_status"]]
        ax3.scatter(row["n_patients"], row["completeness_rate"] * 100, s=row["n_flags"] * 180 + 220,
                    color=c, alpha=0.75, edgecolor="white", linewidth=1.2, zorder=3)
        ax3.annotate(row["site_id"], (row["n_patients"], row["completeness_rate"] * 100),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax3.set_xlabel("Site enrollment (N patients)", fontsize=9)
    ax3.set_ylabel("Data completeness (%)", fontsize=9)
    ax3.set_title("Enrollment vs. Data Completeness by Site", fontsize=12, fontweight="bold")
    handles = [mpatches.Patch(color=COLOR_GOOD, label="OK"),
               mpatches.Patch(color=COLOR_WATCH, label="Watch"),
               mpatches.Patch(color=COLOR_PROBLEM, label="Problem site - review required")]
    ax3.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax3.spines[spine].set_visible(False)

    # --- Panel 4: DQA metrics bar chart by site ---
    ax4 = fig.add_subplot(gs[1, 1])
    dqa_sorted = dqa.sort_values("protocol_deviation_rate", ascending=True)
    bar_colors = [status_color[s] for s in dqa_sorted["dqa_status"]]
    ax4.barh(dqa_sorted["site_id"], dqa_sorted["protocol_deviation_rate"] * 100, color=bar_colors)
    ax4.set_xlabel("Protocol deviation rate (%)", fontsize=9)
    ax4.set_title("Protocol Deviation Rate by Site", fontsize=12, fontweight="bold")
    for i, (site, val) in enumerate(zip(dqa_sorted["site_id"], dqa_sorted["protocol_deviation_rate"] * 100)):
        ax4.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=8)
    problem = dqa_sorted[dqa_sorted["dqa_status"] == "PROBLEM SITE - review required"]
    if len(problem):
        ax4.text(0.98, 0.03, f"Flagged for review: {', '.join(problem['site_id'])}",
                  transform=ax4.transAxes, ha="right", fontsize=8.5, color=COLOR_PROBLEM,
                  fontweight="bold")
    for spine in ("top", "right"):
        ax4.spines[spine].set_visible(False)

    os.makedirs("output/figures", exist_ok=True)
    fig.savefig("output/figures/leadership_dashboard.png", dpi=150)
    plt.close(fig)
    print("Dashboard saved to output/figures/leadership_dashboard.png")


if __name__ == "__main__":
    build_dashboard()
