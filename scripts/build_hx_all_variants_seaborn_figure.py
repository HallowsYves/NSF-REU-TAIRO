"""
All-HX-variants comparison figure, built with seaborn (2026-07-24).

Requested: one figure comparing every HX iteration (HX -> HX2 -> HX3 -> HX4
-> HX5 -> HX6), on both the one condition where the variants actually differ
(grip_state_falsification) and overall pooled performance/safety.

v4 (the common CCAR baseline every HX variant extends) is deliberately NOT
drawn as its own bar (confirmed via sign-off, 2026-07-24) -- this figure's
scope is the HX iteration sequence itself, not a v4-vs-HX comparison (that
comparison already exists in fig_baseline_vs_agnostic_vs_hx.png). v4's
grip_state_falsification rate (14.0%) is still the implicit reference the
significance stars are computed against (every HX evaluation CSV's
`significant_bh` column is "significant vs. v4"), so it's named in the stars'
caption text even without a bar of its own.

The 6 variants are treated as an ORDINAL PROGRESSION (research iteration
order), not an arbitrary categorical set, per the dataviz skill's rule
"sequential/ordinal = one hue, light->dark" -- 6 unrelated categorical hues
here would be harder to read as "this is a version history" than a light-to-
dark ramp is. HX6 is the one adopted variant, so it breaks from the ramp and
keeps the yellow (#eda100) already used for it in
scripts/build_final_hx_figures.py / scripts/build_baseline_comparison_seaborn_figure.py
(same series, same color everywhere in this repo). sac_her (no recovery) is
not a bar either -- it is drawn as a muted dashed reference line, since it
isn't part of the HX iteration sequence at all.

Palette validated (node scripts/validate_palette.js from the dataviz skill):
- HX..HX5 ordinal ramp (5 steps, blue, light->dark) -- ALL CHECKS PASS
  ("#86b6ef,#5598e7,#2a78d6,#1c5cab,#104281" --mode light --ordinal)
- Both hue-boundary pairs (green->HX, HX5->yellow) clear CVD separation and
  the normal-vision floor; the light ramp's early steps and the yellow HX6
  step carry the documented contrast-vs-surface WARN, mitigated the same way
  as the rest of this repo's HX figures: no bar relies on color alone, every
  bar is unambiguously labeled on the x-axis and carries a value/CI, so nothing
  here depends on picking the hue out against the page.

Data provenance (no new evaluation):
- Panel A (grip_state_falsification, the one condition with a real signal):
  read directly, per-variant, from the already-computed n=450 paired-McNemar
  evaluation CSVs -- results/recovery_v4_hx_vs_v4_full_grid.csv (HX, HX2) and
  results/recovery_v4_hx{3,4,5,6}_evaluation.csv (baseline_method ==
  "sac_her_recovery_v4" rows only). sac_her's own rate on this condition
  comes from the "vs_sac_her" rows in the hx6 evaluation CSV (n=450). Wilson
  95% CIs computed fresh via build_recovery_hx_results_package.wilson_ci on
  each rate's k/n, same helper the rest of this repo's HX figures use.
- Panel B (pooled overall success + safety-violation rate): read from the
  "Overall Statistics" table in TAIRO_HX_VARIANT_COMPARISON.md (source:
  final_hx_comparison_summary_table.csv at the time that table was built --
  the checked-in copy of that CSV has since been regenerated down to just
  v2/v3/v4/hx6, so HX/HX2/HX3/HX4/HX5's overall rows are transcribed from the
  committed markdown rather than re-derived here). Hardcoded as a small,
  cited table rather than re-run, matching this repo's "don't recompute what
  already exists" convention for these HX package scripts.

Output:
    results/figures/final_hx_comparison/fig_hx_all_variants_comparison.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from scripts.build_recovery_hx_results_package import wilson_ci

FIG_DIR = "results/figures/final_hx_comparison"

# Iteration order, oldest -> newest. v4 (the common baseline every HX
# variant extends) is intentionally not a series here -- see module
# docstring. sac_her is drawn separately as a reference line.
VARIANT_ORDER = ["HX", "HX2", "HX3", "HX4", "HX5", "HX6"]
VARIANT_LABELS = {
    "HX": "HX\n(stage-gate)",
    "HX2": "HX2\n(+L4 down-weight)",
    "HX3": "HX3\n(+relocalize re-gate)",
    "HX4": "HX4\n(+expert remap)",
    "HX5": "HX5\n(+global fast trigger)",
    "HX6": "HX6\n(+gated fast trigger)",
}
VARIANT_COLORS = {
    "HX":  "#86b6ef",  # blue ramp step 250
    "HX2": "#5598e7",  # blue ramp step 350
    "HX3": "#2a78d6",  # blue ramp step 450
    "HX4": "#1c5cab",  # blue ramp step 550
    "HX5": "#104281",  # blue ramp step 650
    "HX6": "#eda100",  # established yellow -- HX6 is the adopted variant
}

METHOD_NAME = {
    "HX": "sac_her_recovery_v4_hx",
    "HX2": "sac_her_recovery_v4_hx2",
    "HX3": "sac_her_recovery_v4_hx3",
    "HX4": "sac_her_recovery_v4_hx4",
    "HX5": "sac_her_recovery_v4_hx5",
    "HX6": "sac_her_recovery_v4_hx6",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#d8d6cf",
    "grid.color": "#e7e5de",
    "axes.facecolor": "#fcfcfb",
    "figure.facecolor": "#fcfcfb",
    "font.family": "DejaVu Sans",
})


def _savefig(fig, name, pad_inches=0.08):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Panel A data -- grip_state_falsification, the one condition with a signal
# ---------------------------------------------------------------------------

def load_grip_panel() -> pd.DataFrame:
    full_grid = pd.read_csv("results/recovery_v4_hx_vs_v4_full_grid.csv")
    eval_files = {
        "HX3": "results/recovery_v4_hx3_evaluation.csv",
        "HX4": "results/recovery_v4_hx4_evaluation.csv",
        "HX5": "results/recovery_v4_hx5_evaluation.csv",
        "HX6": "results/recovery_v4_hx6_evaluation.csv",
    }

    rows = []

    for variant in ["HX", "HX2"]:
        sub = full_grid[(full_grid["method"] == METHOD_NAME[variant]) &
                         (full_grid["condition"] == "grip_state_falsification")].iloc[0]
        n = int(sub["n_recovery"])
        k = int(round(sub["recovery_success_rate"] * n))
        lo, hi = wilson_ci(k, n)
        rows.append({"variant": variant, "success_rate": sub["recovery_success_rate"],
                      "ci_lo": lo, "ci_hi": hi, "significant_bh": bool(sub["significant_bh"])})

    for variant, path in eval_files.items():
        df = pd.read_csv(path)
        sub = df[(df["method"] == METHOD_NAME[variant]) &
                 (df["condition"] == "grip_state_falsification") &
                 (df["baseline_method"] == "sac_her_recovery_v4")].iloc[0]
        n = int(sub["n_recovery"])
        k = int(round(sub["recovery_success_rate"] * n))
        lo, hi = wilson_ci(k, n)
        rows.append({"variant": variant, "success_rate": sub["recovery_success_rate"],
                      "ci_lo": lo, "ci_hi": hi, "significant_bh": bool(sub["significant_bh"])})

    out = pd.DataFrame(rows)
    out["variant"] = pd.Categorical(out["variant"], categories=VARIANT_ORDER, ordered=True)
    return out.sort_values("variant").reset_index(drop=True)


def load_sac_her_grip_rate() -> float:
    df = pd.read_csv("results/recovery_v4_hx6_evaluation.csv")
    sub = df[(df["condition"] == "grip_state_falsification") &
             (df["comparison"] == "vs_sac_her")].iloc[0]
    return float(sub["baseline_success_rate"])


# ---------------------------------------------------------------------------
# Panel B data -- pooled overall success + safety-violation rate
# (transcribed from TAIRO_HX_VARIANT_COMPARISON.md's "Overall Statistics"
# table -- see module docstring for provenance)
# ---------------------------------------------------------------------------

OVERALL_STATS = pd.DataFrame([
    {"variant": "HX",  "overall_success": 0.3300, "safety_violation": 0.0013},
    {"variant": "HX2", "overall_success": 0.3347, "safety_violation": 0.0009},
    {"variant": "HX3", "overall_success": 0.3362, "safety_violation": 0.0011},
    {"variant": "HX4", "overall_success": 0.3342, "safety_violation": 0.0009},
    {"variant": "HX5", "overall_success": 0.3338, "safety_violation": 0.0011},
    {"variant": "HX6", "overall_success": 0.3347, "safety_violation": 0.0006},
])
OVERALL_STATS["variant"] = pd.Categorical(OVERALL_STATS["variant"], categories=VARIANT_ORDER, ordered=True)
OVERALL_STATS = OVERALL_STATS.sort_values("variant").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

SHORT_LABELS = {
    "HX": "HX", "HX2": "HX2", "HX3": "HX3",
    "HX4": "HX4", "HX5": "HX5", "HX6": "HX6",
}


def _bar(ax, df, x_col, y_col, order, ref_line=None, ref_label=None,
         pct=True, star_col=None, short_labels=False, rotate=0):
    label_map = SHORT_LABELS if short_labels else VARIANT_LABELS
    labels = [label_map[v] for v in order]
    colors = [VARIANT_COLORS[v] for v in order]
    sub = df.set_index(x_col).loc[order].reset_index()

    bars = ax.bar(labels, sub[y_col], color=colors, edgecolor=INK_PRIMARY, linewidth=0.6, zorder=3)

    if "ci_lo" in sub.columns:
        yerr_lo = (sub[y_col] - sub["ci_lo"]).clip(lower=0).to_numpy()
        yerr_hi = (sub["ci_hi"] - sub[y_col]).clip(lower=0).to_numpy()
        ax.errorbar(labels, sub[y_col], yerr=[yerr_lo, yerr_hi], fmt="none",
                     ecolor=INK_PRIMARY, elinewidth=1.0, capsize=3, alpha=0.75, zorder=5)

    if star_col is not None:
        for bar, sig, yhi in zip(bars, sub[star_col], sub[y_col] if "ci_hi" not in sub.columns else sub["ci_hi"]):
            if sig:
                ax.text(bar.get_x() + bar.get_width() / 2, yhi + 0.012, "*",
                        ha="center", va="bottom", fontsize=24, color=INK_PRIMARY, fontweight="bold")

    if ref_line is not None:
        ax.axhline(ref_line, color=INK_MUTED, linewidth=1.3, linestyle="--", zorder=2)
        # Stacked under the "* BH-significant..." caption (same left anchor,
        # one line down), not next to the dashed line itself and not beside
        # that caption -- at the larger poster font size and this column's
        # narrow width, either placement collides: next to the line hits the
        # error bars/stars, and side-by-side-at-the-top text is wide enough
        # to overlap the other caption.
        ax.text(0.01, 0.90, ref_label, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, color=INK_SECONDARY, style="italic")

    if pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=16)
    ax.tick_params(axis="x", colors=INK_PRIMARY, labelsize=15)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right", rotation_mode="anchor")
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=True)


def _draw_all_panels(ax_grip, ax_succ, ax_safe, grip_df, sac_her_grip) -> None:
    # short_labels=True (not the descriptive "(+L4 down-weight)" etc. labels)
    # in both layouts now -- at the larger poster font size, the two-line
    # descriptive labels are wide enough to overlap their neighbors even in
    # the landscape layout. The mechanism names belong in the caption text,
    # not packed into the x-axis, matching how Fig 2's HX/HX2/... labels
    # already work.
    _bar(ax_grip, grip_df, "variant", "success_rate", VARIANT_ORDER,
         ref_line=sac_her_grip, ref_label=f"sac_her, no recovery ({sac_her_grip:.1%})",
         star_col="significant_bh", short_labels=True)
    # Extra headroom above the highest error bar/star (~0.24) so the two
    # stacked caption lines (axes-fraction y=0.90/0.99) have clear space and
    # don't overlap the significance stars at the larger poster font size.
    ax_grip.set_ylim(0, 0.42)
    ax_grip.set_ylabel("Success rate\n(grip_state_falsification)", fontsize=17, color=INK_PRIMARY)
    ax_grip.text(0.01, 0.99, "* BH-significant improvement vs. v4 (p<0.05)",
                 transform=ax_grip.transAxes, ha="left", va="top", fontsize=14,
                 color=INK_MUTED, style="italic")

    _bar(ax_succ, OVERALL_STATS, "variant", "overall_success", VARIANT_ORDER, short_labels=True)
    ax_succ.set_ylim(0.30, 0.35)
    ax_succ.set_ylabel("Overall success rate\n(pooled, all 11 conditions)", fontsize=16, color=INK_PRIMARY)

    _bar(ax_safe, OVERALL_STATS, "variant", "safety_violation", VARIANT_ORDER, short_labels=True)
    ax_safe.set_ylim(0, 0.0016)
    ax_safe.set_ylabel("Safety violation rate\n(pooled, all 11 conditions)", fontsize=16, color=INK_PRIMARY)


def main() -> None:
    grip_df = load_grip_panel()
    sac_her_grip = load_sac_her_grip_rate()

    # Landscape, 2x2 grid (grip panel spans the top row) -- for full-width use.
    # Wider figure + more wspace than before -- at the larger poster font
    # size, the safety-violation panel's y-axis label was wide enough to
    # bleed into the success-rate panel next to it.
    fig = plt.figure(figsize=(15.0, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.45, wspace=0.45)
    ax_grip = fig.add_subplot(gs[0, :])
    ax_succ = fig.add_subplot(gs[1, 0])
    ax_safe = fig.add_subplot(gs[1, 1])
    _draw_all_panels(ax_grip, ax_succ, ax_safe, grip_df, sac_her_grip)
    _savefig(fig, "fig_hx_all_variants_comparison.png", pad_inches=0.15)

    # Portrait, 3 panels stacked -- for a narrow poster column, matching the
    # same "_vertical" treatment already used for the latency figure
    # (scripts/build_hx_all_variants_latency_figure.py). Each of the three
    # charts now gets the figure's FULL width instead of half (succ/safe
    # were side by side in the landscape version), consistent with how this
    # panel sits alongside the other two (already-vertical) poster panels.
    fig_v = plt.figure(figsize=(6.5, 12.5))
    gs_v = fig_v.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 1.0], hspace=0.5)
    ax_grip_v = fig_v.add_subplot(gs_v[0])
    ax_succ_v = fig_v.add_subplot(gs_v[1])
    ax_safe_v = fig_v.add_subplot(gs_v[2])
    _draw_all_panels(ax_grip_v, ax_succ_v, ax_safe_v, grip_df, sac_her_grip)
    _savefig(fig_v, "fig_hx_all_variants_comparison_vertical.png", pad_inches=0.15)


if __name__ == "__main__":
    main()
