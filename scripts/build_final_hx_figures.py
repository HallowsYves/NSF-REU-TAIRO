"""
Final tables + charts for the mentor's final-push checklist (2026-07-22,
updated 2026-07-23 for the hx6 adoption).

Extends scripts/build_recovery_hx_results_package.py's 3-method headline
figure to the mentor's requested 4-arm comparison (no recovery / earlier
baselines v2+v3 / gradual-response v4 / final selective v4-HX6) and adds the
recovery-latency figure the checklist explicitly asks for -- not previously
built anywhere in this repo. Reuses that script's validated helpers
(wilson_ci, _df_to_markdown, condition-label map, INK/SURFACE/GRIDLINE
styling constants) rather than redefining them.

Inputs (already built by scripts/build_final_hx_comparison.py):
    results/final_hx_comparison_success_safety.csv
    results/final_hx_comparison_timing.csv
    results/final_hx_comparison_final_head_to_head.csv

Outputs:
    results/figures/final_hx_comparison/fig1_success_by_condition.png
    results/figures/final_hx_comparison/fig2_recovery_latency.png
    results/final_hx_comparison_summary_table.{csv,md}
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

from config import ALL_CONDITIONS
from scripts.build_recovery_hx_results_package import (
    COND_LABELS, INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE, GRIDLINE,
    _cond_label, _df_to_markdown, wilson_ci,
)
from scripts.build_final_hx_comparison import ARMS, load_episode_data

FIG_DIR = "results/figures/final_hx_comparison"

# Validated (node scripts/validate_palette.js, light mode -- ALL CHECKS PASS,
# yellow slot 4 carries the documented contrast WARN, mitigated the same way
# as build_recovery_hx_results_package.py: no per-bar labels relying on
# color alone, companion table + legend carry the exact values instead).
# slot 1/2/4 match that script's existing sac_her/v4/v4_hx2 colors exactly
# (same series, same color everywhere in the repo -- v4_hx6 inherits v4_hx2's
# yellow slot 4 since it superseded it as the same "final selective" series,
# not a new one); slots 6/7 (orange, violet) are new -- v2/v3 were never
# charted in this palette before.
METHOD_COLORS = {
    "sac_her":                 "#2a78d6",  # slot 1 blue
    "sac_her_recovery_v2":     "#eb6834",  # slot 6 orange
    "sac_her_recovery_v3":     "#4a3aa7",  # slot 7 violet
    "sac_her_recovery_v4":     "#008300",  # slot 2 green
    "sac_her_recovery_v4_hx6": "#eda100",  # slot 4 yellow
}
METHOD_LABELS = {
    "sac_her":                 "No recovery (SAC+HER)",
    "sac_her_recovery_v2":     "Recovery v2 (earlier baseline)",
    "sac_her_recovery_v3":     "Recovery v3 (earlier baseline)",
    "sac_her_recovery_v4":     "Recovery v4 (gradual-response CCAR)",
    "sac_her_recovery_v4_hx6": "Recovery v4-HX6 (final, selective)",
}
METHOD_ORDER = list(ARMS.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _savefig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[final-figs] -> {path}")


# ---------------------------------------------------------------------------
# Table: success rate + Wilson CI, method x condition (5 arms)
# ---------------------------------------------------------------------------

def build_success_table(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for condition in ALL_CONDITIONS:
            sub = combined[(combined["method"] == method) & (combined["condition"] == condition)]
            n = len(sub)
            if n == 0:
                continue
            k = int(sub["success"].sum())
            rate = k / n
            lo, hi = wilson_ci(k, n)
            rows.append({
                "method": method, "condition": condition, "n": n,
                "success_rate": rate, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fig 1: 5-arm grouped bar, success rate by condition
# ---------------------------------------------------------------------------

def fig1_success_by_condition(table: pd.DataFrame) -> None:
    conditions = ALL_CONDITIONS
    x = np.arange(len(conditions))
    width = 0.16

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, method in enumerate(METHOD_ORDER):
        sub = table[table["method"] == method].set_index("condition")
        rates = [sub.loc[c, "success_rate"] if c in sub.index else np.nan for c in conditions]
        los = [max(0.0, sub.loc[c, "success_rate"] - sub.loc[c, "ci_lo"]) if c in sub.index else 0 for c in conditions]
        his = [max(0.0, sub.loc[c, "ci_hi"] - sub.loc[c, "success_rate"]) if c in sub.index else 0 for c in conditions]
        offset = (i - 2) * width
        ax.bar(
            x + offset, rates, width,
            yerr=[los, his], capsize=2, error_kw={"linewidth": 0.8, "ecolor": INK_SECONDARY},
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            alpha=0.92, edgecolor=SURFACE, linewidth=0.4, zorder=3,
        )
    # No per-bar value labels -- 5 close-valued series per group collide;
    # exact values live in the companion table (relief rule for the yellow
    # series, same mitigation as build_recovery_hx_results_package.py).

    ax.set_xticks(x)
    ax.set_xticklabels([_cond_label(c) for c in conditions], fontsize=8, color=INK_PRIMARY)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Task success rate", fontsize=10, color=INK_PRIMARY)
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")

    ax.set_title(
        "Fig 1 — PickAndPlace success rate by attack condition (n=450/cell)\n"
        "No recovery vs. earlier baselines (v2/v3) vs. gradual-response (v4) vs. final selective (v4-HX6)",
        fontsize=12, color=INK_PRIMARY, loc="left", pad=12,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=1)
    _savefig(fig, "fig1_success_by_condition.png")


# ---------------------------------------------------------------------------
# Fig 2: recovery-latency figure (mentor-requested, new)
# ---------------------------------------------------------------------------

def fig2_recovery_latency(delay_df: pd.DataFrame, timing_df: pd.DataFrame) -> None:
    """Stacked horizontal bar: detection delay + recovery-response delay =
    total steps from attack onset to full/meaningful recovery authority,
    per method (mean across all non-clean conditions, since detection/
    response timing is a controller property, not a per-condition one --
    the per-condition breakdown lives in the companion CSV). Detection/
    response delay are plain descriptive means (no sac_her baseline exists
    for either -- it never triggers recovery at all, see
    build_final_hx_comparison.py's delay_df). recovery_time (total active
    steps, still a meaningful vs-sac_her comparison since sac_her's is
    always exactly 0) is annotated alongside as a distinct property -- a
    method can respond fast but stay active only briefly (v2/v3, hard
    override + early-exit) or respond slowly but then blend for most of
    the episode (v4/v4-HX6's continuous ramp) -- collapsing the two into
    one number would hide exactly the finding this figure exists to show.
    """
    methods = [m for m in METHOD_ORDER if m != "sac_her"]
    detection = []
    response = []
    rec_time = []
    for m in methods:
        sub = delay_df[delay_df.method == m]
        detection.append(sub["detection_delay_mean"].mean() if len(sub) else np.nan)
        response.append(sub["response_delay_mean"].mean() if len(sub) else np.nan)
        rt = timing_df[(timing_df.method == m) & (timing_df.metric == "recovery_time")]["recovery_mean"]
        rec_time.append(rt.mean() if len(rt) else np.nan)

    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    colors = [METHOD_COLORS[m] for m in methods]
    ax.barh(y, detection, color=colors, alpha=0.55, edgecolor=SURFACE,
             linewidth=0.5, zorder=3, label="Detection delay (onset → first trigger)")
    ax.barh(y, response, left=detection, color=colors, alpha=0.95, edgecolor=SURFACE,
             linewidth=0.5, zorder=3, label="Recovery-response delay (trigger → full authority)")

    for i, (d, r, rt) in enumerate(zip(detection, response, rec_time)):
        total = d + (r if not np.isnan(r) else 0)
        ax.text(total + 1.5, i, f"total {total:.0f} steps · active {rt:.0f} steps",
                 va="center", fontsize=8.5, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=9.5, color=INK_PRIMARY)
    ax.set_xlabel("Steps (mean across all 10 non-clean conditions)", fontsize=10, color=INK_PRIMARY)
    ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.set_xlim(0, max(d + (r if not np.isnan(r) else 0) for d, r in zip(detection, response)) * 1.35)

    ax.set_title(
        "Fig 2 — Recovery latency: detection + response delay by method\n"
        "Earlier baselines (v2/v3) trigger fast and respond instantly (hard override); "
        "v4/v4-HX6 detect more slowly and ramp gradually (continuous blend)",
        fontsize=11.5, color=INK_PRIMARY, loc="left", pad=12,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=INK_MUTED, alpha=0.55),
        plt.Rectangle((0, 0), 1, 1, color=INK_MUTED, alpha=0.95),
    ]
    ax.legend(handles, ["Detection delay", "Recovery-response delay"],
              loc="lower right", fontsize=8.5, frameon=False)
    _savefig(fig, "fig2_recovery_latency.png")


# ---------------------------------------------------------------------------
# Summary table: all 8 mentor-requested metrics, one row per method
# ---------------------------------------------------------------------------

def build_summary_table(combined: pd.DataFrame, delay_df: pd.DataFrame,
                          timing_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        ep = combined[combined.method == method]
        clean_ep = ep[ep.condition == "clean"]
        row = {
            "method": method,
            "label": METHOD_LABELS[method],
            "task_success_rate_overall": ep["success"].mean(),
            "clean_task_performance": clean_ep["success"].mean() if len(clean_ep) else np.nan,
            "safety_violation_rate": ep["safety_violation"].mean(),
        }
        dsub = delay_df[delay_df.method == method]
        row["detection_delay_steps"] = dsub["detection_delay_mean"].mean() if len(dsub) else np.nan
        row["recovery_response_delay_steps"] = dsub["response_delay_mean"].mean() if len(dsub) else np.nan
        for metric_col, out_col in [
            ("recovery_time", "recovery_time_steps"),
            ("num_interventions", "num_interventions"),
        ]:
            sub = timing_df[(timing_df.method == method) & (timing_df.metric == metric_col)]
            row[out_col] = sub["recovery_mean"].mean() if len(sub) else np.nan
        overhead = timing_df[
            (timing_df.method == method) & (timing_df.metric == "completion_time_overhead")
        ]
        row["completion_time_overhead_steps"] = overhead["delta"].mean() if len(overhead) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    combined = load_episode_data()
    print(f"[final-figs] Loaded {len(combined)} episodes.")

    success_table = build_success_table(combined)
    fig1_success_by_condition(success_table)

    delay_df = pd.read_csv("results/final_hx_comparison_delays.csv")
    timing_df = pd.read_csv("results/final_hx_comparison_timing.csv")
    fig2_recovery_latency(delay_df, timing_df)

    summary = build_summary_table(combined, delay_df, timing_df)

    out_csv = "results/final_hx_comparison_summary_table.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[final-figs] -> {out_csv}")

    disp = summary.drop(columns=["method"]).set_index("label")
    disp = disp.map(lambda v: f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else "—")
    out_md = "results/final_hx_comparison_summary_table.md"
    with open(out_md, "w") as f:
        f.write("# Final TAIRO-HX comparison — all 8 mentor-requested metrics\n\n")
        f.write("(n=450/method, seeds 0-14, all 11 conditions, clean_2M checkpoint. "
                "Task-success/clean/safety are rates over all episodes; timing metrics "
                "are means across non-clean conditions/episodes where recovery triggered "
                "at least once; completion-time overhead is successful episodes only.)\n\n")
        f.write(_df_to_markdown(disp, index=True))
        f.write("\n")
    print(f"[final-figs] -> {out_md}")


if __name__ == "__main__":
    main()
