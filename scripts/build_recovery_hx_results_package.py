"""
Poster/video results package for the TAIRO-HX recovery-integration work.

Mentor's explicit ask (2026-07-22): present the stage-aware (v4_hx),
attack-aware (v4_hx2), and routing-fix (v4_hx3) recovery results with
tables, charts, statistical comparisons, and brief interpretations. This
is packaging/presentation work over already-validated data -- no new
experiments, no new statistics. Every number here is read from CSVs
already produced and reviewed this week:

    results/recovery_do_no_harm_audit.csv        (v2/v3/v4/hx/hx2 vs sac_her, 55 comparisons)
    results/recovery_v4_hx_vs_v4_full_grid.csv   (hx vs v4, hx2 vs v4, 22 comparisons)
    results/recovery_v4_hx3_evaluation.csv       (hx3 vs v4/hx2, hx3 vs sac_her, 44 comparisons)

Per-condition success rates (Fig 1, Table 1) are computed directly from the
underlying episode data via scripts/evaluate_recovery_v4_hx3.load_all_episodes(),
which already knows how to combine every data source at full power (n=450)
and dedupe correctly -- reused as-is, not reimplemented.

Outputs:
    results/figures/recovery_hx_package/fig1_headline_success_by_condition.png
    results/figures/recovery_hx_package/fig2_do_no_harm_forest.png
    results/figures/recovery_hx_package/fig3_hx_variant_forest.png
    results/recovery_hx_success_by_method_condition.csv  (+ .md)
    results/recovery_hx_key_findings_table.csv           (+ .md)
    results/recovery_hx_results_summary.md               (interpretive report)
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
from matplotlib.lines import Line2D

from config import ALL_CONDITIONS
from scripts.evaluate_recovery_v4_hx3 import load_all_episodes
from scripts.build_figures import CONDITION_LABELS, _PICKANDPLACE_COND_LABELS

FIG_DIR = "results/figures/recovery_hx_package"
COND_LABELS = {**CONDITION_LABELS, **_PICKANDPLACE_COND_LABELS}

# ---------------------------------------------------------------------------
# Fixed identity colors -- validated dataviz palette, slots 1-5, adjacent-pair
# and all-pairs CVD/normal-vision checks both pass (relief rule satisfied via
# the direct value labels every bar/point already carries in this script).
# Never repainted per-figure: same method always gets the same color.
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    "sac_her":                 "#2a78d6",  # slot 1 blue
    "sac_her_recovery_v4":     "#008300",  # slot 2 green
    "sac_her_recovery_v4_hx":  "#e87ba4",  # slot 3 magenta
    "sac_her_recovery_v4_hx2": "#eda100",  # slot 4 yellow
    "sac_her_recovery_v4_hx3": "#1baf7a",  # slot 5 aqua
}
METHOD_LABELS = {
    "sac_her":                 "SAC+HER (no recovery)",
    "sac_her_recovery_v4":     "Recovery v4 (CCAR)",
    "sac_her_recovery_v4_hx":  "v4-HX (Level 1 stage-gate) — not adopted",
    "sac_her_recovery_v4_hx2": "v4-HX2 (Level 1 + Level 4) — adopted",
    "sac_her_recovery_v4_hx3": "v4-HX3 (Level 4 re-gate) — null result",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _cond_label(c: str) -> str:
    return COND_LABELS.get(c, c.replace("_", "\n"))


def _df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal GitHub-flavored markdown table writer (no `tabulate` dependency)."""
    cols = ([df.index.name or ""] if index else []) + list(df.columns.astype(str))
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else []) + [str(v) for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return ((center - half) / denom, (center + half) / denom)


def _savefig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[hx-package] -> {path}")


# ---------------------------------------------------------------------------
# Table 1: success rate + Wilson 95% CI, method x condition
# ---------------------------------------------------------------------------

def build_success_table(combined: pd.DataFrame) -> pd.DataFrame:
    methods = list(METHOD_COLORS.keys())
    rows = []
    for method in methods:
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


def write_success_table(table: pd.DataFrame) -> None:
    out_csv = "results/recovery_hx_success_by_method_condition.csv"
    table.to_csv(out_csv, index=False)
    print(f"[hx-package] -> {out_csv}")

    pivot = table.pivot(index="condition", columns="method", values="success_rate")
    pivot = pivot.reindex(ALL_CONDITIONS)
    pivot = pivot[[m for m in METHOD_COLORS if m in pivot.columns]]
    pivot.columns = [METHOD_LABELS[m] for m in pivot.columns]
    pivot.index = [_cond_label(c).replace("\n", " ") for c in pivot.index]

    out_md = "results/recovery_hx_success_by_method_condition.md"
    disp = pivot.map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    disp.index.name = "condition"
    with open(out_md, "w") as f:
        f.write("# Success rate by method x condition (n=450/cell unless noted)\n\n")
        f.write(_df_to_markdown(disp, index=True))
        f.write("\n")
    print(f"[hx-package] -> {out_md}")


# ---------------------------------------------------------------------------
# Fig 1: headline grouped bar -- sac_her vs v4 vs v4_hx2, all 11 conditions
# ---------------------------------------------------------------------------

def fig1_headline(table: pd.DataFrame) -> None:
    methods = ["sac_her", "sac_her_recovery_v4", "sac_her_recovery_v4_hx2"]
    conditions = ALL_CONDITIONS
    x = np.arange(len(conditions))
    width = 0.27

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, method in enumerate(methods):
        sub = table[table["method"] == method].set_index("condition")
        rates = [sub.loc[c, "success_rate"] if c in sub.index else np.nan for c in conditions]
        los = [max(0.0, sub.loc[c, "success_rate"] - sub.loc[c, "ci_lo"]) if c in sub.index else 0 for c in conditions]
        his = [max(0.0, sub.loc[c, "ci_hi"] - sub.loc[c, "success_rate"]) if c in sub.index else 0 for c in conditions]
        offset = (i - 1) * width
        ax.bar(
            x + offset, rates, width,
            yerr=[los, his], capsize=2.5, error_kw={"linewidth": 1, "ecolor": INK_SECONDARY},
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            alpha=0.92, edgecolor=SURFACE, linewidth=0.5, zorder=3,
        )
    # No per-bar value labels -- with 3 close-valued series per condition group,
    # labels collide (e.g. object_pose_spoof's 25-28% cluster). Exact values live
    # in the companion table (results/recovery_hx_success_by_method_condition.md),
    # which also satisfies the dataviz "relief rule" for the yellow series.

    ax.set_xticks(x)
    ax.set_xticklabels([_cond_label(c) for c in conditions], fontsize=8.5, color=INK_PRIMARY)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Task success rate", fontsize=10, color=INK_PRIMARY)
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")

    ax.set_title(
        "Fig 1 — PickAndPlace success rate by attack condition\n"
        "No recovery vs. Recovery v4 (paper's current B4) vs. Recovery v4-HX2 (adopted extension)",
        fontsize=12, color=INK_PRIMARY, loc="left", pad=12,
    )
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.text(0.006, 0.005,
              "clean_2M checkpoint. n=450/condition (seeds 0-14 x 30 episodes). "
              "Error bars: 95% Wilson score interval.",
              fontsize=7.5, color=INK_MUTED)

    _savefig(fig, "fig1_headline_success_by_condition.png")


# ---------------------------------------------------------------------------
# Forest-plot helper
# ---------------------------------------------------------------------------

def _forest_plot(rows_by_method: dict, conditions: list, title: str, subtitle: str,
                  out_name: str, xlabel: str) -> None:
    """rows_by_method: {method: DataFrame indexed by condition with delta/ci_lo/ci_hi/significant_bh}."""
    n_methods = len(rows_by_method)
    n_cond = len(conditions)
    fig_h = max(4.5, 0.62 * n_cond + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y_base = np.arange(n_cond)[::-1]  # first condition at top
    row_span = 0.62
    offsets = np.linspace(-row_span / 2, row_span / 2, n_methods) if n_methods > 1 else [0]

    for (method, df), off in zip(rows_by_method.items(), offsets):
        color = METHOD_COLORS[method]
        ys = []
        for yb, cond in zip(y_base, conditions):
            if cond not in df.index:
                continue
            r = df.loc[cond]
            y = yb + off
            sig = bool(r["significant_bh"])
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color,
                    linewidth=1.6 if sig else 1.1, alpha=1.0 if sig else 0.45, zorder=2)
            ax.plot(r["delta"], y,
                    marker="o", markersize=6.5 if sig else 5,
                    markerfacecolor=color if sig else SURFACE,
                    markeredgecolor=color, markeredgewidth=1.4,
                    alpha=1.0 if sig else 0.55, zorder=3)
            if sig:
                ax.annotate(f"{r['delta']:+.1%}", (r["ci_hi"], y),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=7.5, color=INK_PRIMARY, va="center", fontweight="bold")
            ys.append(y)

    ax.axvline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y_base)
    ax.set_yticklabels([_cond_label(c).replace("\n", " ") for c in conditions], fontsize=9.5, color=INK_PRIMARY)
    ax.set_xlabel(xlabel, fontsize=10, color=INK_PRIMARY)
    ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")

    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker="o", markerfacecolor=METHOD_COLORS[m],
               markeredgecolor=METHOD_COLORS[m], linewidth=1.6, markersize=6.5, label=METHOD_LABELS[m])
        for m in rows_by_method
    ]
    legend_handles.append(
        Line2D([0], [0], color=INK_MUTED, marker="o", markerfacecolor=SURFACE,
               markeredgecolor=INK_MUTED, linewidth=1.1, markersize=5, alpha=0.6,
               label="filled marker + bold label = BH-significant; open/faint = not significant")
    )
    # Legend below the axes, out of the data area entirely -- every row has a
    # point near x=0, so any in-plot corner risks sitting on top of a marker.
    ax.legend(handles=legend_handles, loc="upper center", fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.11 - 0.006 * n_cond), ncol=1)

    ax.set_title(f"{title}\n{subtitle}", fontsize=12, color=INK_PRIMARY, loc="left", pad=12)
    fig.subplots_adjust(bottom=0.14 + 0.012 * n_methods)

    _savefig(fig, out_name)


# ---------------------------------------------------------------------------
# Fig 2: do-no-harm forest -- v4 and v4_hx2 vs sac_her, all 11 conditions
# ---------------------------------------------------------------------------

def fig2_do_no_harm() -> None:
    audit = pd.read_csv("results/recovery_do_no_harm_audit.csv")
    rows_by_method = {}
    for method in ["sac_her_recovery_v4", "sac_her_recovery_v4_hx2"]:
        sub = audit[audit["method"] == method].set_index("condition")
        rows_by_method[method] = sub
    _forest_plot(
        rows_by_method, ALL_CONDITIONS,
        "Fig 2 — Do-no-harm audit: recovery vs. no recovery at all",
        "Δ success rate = method − sac_her, 95% CI, Benjamini-Hochberg FDR-corrected (55-comparison grid)",
        "fig2_do_no_harm_forest.png",
        "Δ success rate vs. sac_her (positive = recovery helps)",
    )


# ---------------------------------------------------------------------------
# Fig 3: hx-variant forest -- hx, hx2, hx3 vs plain v4, all 11 conditions
# ---------------------------------------------------------------------------

def fig3_hx_variants() -> None:
    grid = pd.read_csv("results/recovery_v4_hx_vs_v4_full_grid.csv")
    hx3 = pd.read_csv("results/recovery_v4_hx3_evaluation.csv")
    hx3_vs_v4 = hx3[(hx3["comparison"] == "vs_v4_or_hx2") &
                    (hx3["baseline_method"] == "sac_her_recovery_v4")]

    rows_by_method = {
        "sac_her_recovery_v4_hx":  grid[grid["method"] == "sac_her_recovery_v4_hx"].set_index("condition"),
        "sac_her_recovery_v4_hx2": grid[grid["method"] == "sac_her_recovery_v4_hx2"].set_index("condition"),
        "sac_her_recovery_v4_hx3": hx3_vs_v4.set_index("condition"),
    }
    _forest_plot(
        rows_by_method, ALL_CONDITIONS,
        "Fig 3 — HX-variant investigation: each variant vs. plain Recovery v4",
        "Δ success rate = variant − v4, 95% CI, BH-corrected. Only hx2's grip_state_falsification win is confirmed.",
        "fig3_hx_variant_forest.png",
        "Δ success rate vs. plain Recovery v4",
    )


# ---------------------------------------------------------------------------
# Table 2: curated key-findings table
# ---------------------------------------------------------------------------

def build_key_findings_table() -> pd.DataFrame:
    audit = pd.read_csv("results/recovery_do_no_harm_audit.csv")
    grid = pd.read_csv("results/recovery_v4_hx_vs_v4_full_grid.csv")
    hx3 = pd.read_csv("results/recovery_v4_hx3_evaluation.csv")

    def _row(df, method, baseline, condition, label):
        m = df[(df["method"] == method) & (df["baseline_method"] == baseline) & (df["condition"] == condition)]
        if m.empty:
            return None
        r = m.iloc[0]
        return {
            "finding": label,
            "condition": condition,
            "method": method,
            "baseline": baseline,
            "delta": r["delta"],
            "ci_lo": r["ci_lo"],
            "ci_hi": r["ci_hi"],
            "p_value_bh": r["p_value_bh"],
            "significant_bh": bool(r["significant_bh"]),
        }

    rows = [
        _row(audit, "sac_her_recovery_v4", "sac_her", "grip_state_falsification",
             "Plain v4 does significant HARM vs. no recovery"),
        _row(grid, "sac_her_recovery_v4_hx2", "sac_her_recovery_v4", "grip_state_falsification",
             "v4-HX2 fixes the harm: significant win vs. plain v4"),
        _row(audit, "sac_her_recovery_v4_hx2", "sac_her", "grip_state_falsification",
             "v4-HX2 vs. no recovery: parity restored (not significant either way)"),
        _row(grid, "sac_her_recovery_v4_hx", "sac_her_recovery_v4", "object_pose_spoof",
             "v4-HX alone: object_pose_spoof regression, no longer significant at full grid"),
        _row(hx3[hx3["comparison"] == "vs_v4_or_hx2"], "sac_her_recovery_v4_hx3",
             "sac_her_recovery_v4", "object_pose_spoof",
             "v4-HX3 targeted fix vs. plain v4: null result"),
        _row(hx3[hx3["comparison"] == "vs_v4_or_hx2"], "sac_her_recovery_v4_hx3",
             "sac_her_recovery_v4_hx2", "object_pose_spoof",
             "v4-HX3 vs. adopted v4-HX2: null result"),
        _row(hx3[hx3["comparison"] == "vs_v4_or_hx2"], "sac_her_recovery_v4_hx3",
             "sac_her_recovery_v4", "grip_state_falsification",
             "v4-HX3 preserves the hx2 grip_state_falsification win vs. plain v4"),
    ]
    return pd.DataFrame([r for r in rows if r is not None])


def write_key_findings_table(table: pd.DataFrame) -> None:
    out_csv = "results/recovery_hx_key_findings_table.csv"
    table.to_csv(out_csv, index=False)
    print(f"[hx-package] -> {out_csv}")

    disp = table.copy()
    disp["delta"] = disp["delta"].map(lambda v: f"{v:+.1%}")
    disp["95% CI"] = table.apply(lambda r: f"[{r['ci_lo']:+.1%}, {r['ci_hi']:+.1%}]", axis=1)
    disp["p_bh"] = table["p_value_bh"].map(lambda v: f"{v:.4f}" if v >= 0.0001 else f"{v:.1e}")
    disp["significant"] = table["significant_bh"].map(lambda v: "YES" if v else "no")
    disp = disp[["finding", "condition", "delta", "95% CI", "p_bh", "significant"]]

    out_md = "results/recovery_hx_key_findings_table.md"
    with open(out_md, "w") as f:
        f.write("# Key statistical findings — Recovery v4 HX variants\n\n")
        f.write(_df_to_markdown(disp, index=False))
        f.write("\n")
    print(f"[hx-package] -> {out_md}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    combined = load_all_episodes()
    print(f"[hx-package] Loaded {len(combined)} episodes.")

    success_table = build_success_table(combined)
    write_success_table(success_table)

    fig1_headline(success_table)
    fig2_do_no_harm()
    fig3_hx_variants()

    findings_table = build_key_findings_table()
    write_key_findings_table(findings_table)

    print(f"\n[hx-package] Done. Figures in {FIG_DIR}/, tables in results/.")


if __name__ == "__main__":
    main()
