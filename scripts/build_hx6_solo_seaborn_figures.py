"""
Standalone Recovery v4-HX6 figures, built with seaborn (2026-07-23).

Scope, per explicit ask: hx6 alone, no comparison against sac_her/v2/v3/v4/
earlier hx variants. Every number is read straight from the already-built
final-hx-comparison CSVs -- no new experiments, no recomputation.

Inputs:
    results/final_hx_comparison_success_safety.csv
    results/final_hx_comparison_delays.csv

Outputs:
    results/figures/hx6_solo/fig1_success_by_condition.png
    results/figures/hx6_solo/fig2_recovery_latency.png
    results/figures/hx6_solo/fig3_evaluation_table.png
    results/hx6_solo_evaluation_table.csv
    results/hx6_solo_evaluation_table.md
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

from config import ALL_CONDITIONS
from scripts.build_final_hx_figures import METHOD_COLORS, METHOD_LABELS
from scripts.build_recovery_hx_results_package import COND_LABELS, _cond_label, _df_to_markdown

METHOD = "sac_her_recovery_v4_hx6"
HX6_COLOR = METHOD_COLORS[METHOD]          # canonical repo color for this series (#eda100)
HX6_LABEL = METHOD_LABELS[METHOD]

FIG_DIR = "results/figures/hx6_solo"
SUCCESS_SAFETY_CSV = "results/final_hx_comparison_success_safety.csv"
DELAYS_CSV = "results/final_hx_comparison_delays.csv"

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#d8d6cf",
    "grid.color": "#e7e5de",
    "axes.facecolor": "#fcfcfb",
    "figure.facecolor": "#fcfcfb",
    "font.family": "DejaVu Sans",
})
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def _savefig(fig, name, pad_inches=0.08):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Load hx6-only data
# ---------------------------------------------------------------------------

def load_success_by_condition() -> pd.DataFrame:
    df = pd.read_csv(SUCCESS_SAFETY_CSV)
    df = df[(df["method"] == METHOD) & (df["metric"] == "task_success_rate")].copy()
    df["condition"] = pd.Categorical(df["condition"], categories=ALL_CONDITIONS, ordered=True)
    df = df.sort_values("condition")
    df["success_rate"] = df["recovery_success_rate"].astype(float)
    return df[["condition", "success_rate"]]


def load_safety_by_condition() -> pd.DataFrame:
    df = pd.read_csv(SUCCESS_SAFETY_CSV)
    df = df[(df["method"] == METHOD) & (df["metric"] == "safety_violation")].copy()
    df["safety_violation_rate"] = df["recovery_mean"].astype(float)
    return df[["condition", "safety_violation_rate"]]


def load_latency_by_condition() -> pd.DataFrame:
    df = pd.read_csv(DELAYS_CSV)
    df = df[df["method"] == METHOD].copy()
    keep = ["condition", "trigger_rate", "detection_delay_mean", "detection_delay_median",
            "response_delay_mean", "response_delay_median"]
    return df[keep]


# ---------------------------------------------------------------------------
# Fig 1 -- success rate by condition
# ---------------------------------------------------------------------------

def build_fig1(success_df: pd.DataFrame) -> None:
    # .astype(str) strips the ordered-Categorical dtype on "condition" --
    # without this, seaborn draws y-ticks in the Categorical's fixed
    # taxonomy order (from ALL_CONDITIONS) instead of the row order of the
    # sorted dataframe, silently pairing each bar's length with the wrong
    # condition label.
    plot_df = success_df.sort_values("success_rate", ascending=False).copy()
    plot_df["condition"] = plot_df["condition"].astype(str)
    plot_df["label"] = plot_df["condition"].map(lambda c: _cond_label(c).replace("\n", " "))
    order = plot_df["label"].tolist()

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    sns.barplot(
        data=plot_df, x="label", y="success_rate", order=order,
        color=HX6_COLOR, edgecolor=INK_PRIMARY, linewidth=0.6, ax=ax,
    )
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(i, row["success_rate"] + 0.015, f"{row['success_rate']*100:.1f}%",
                ha="center", fontsize=9.5, color=INK_PRIMARY, fontweight="bold")

    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Task success rate", fontsize=10.5, color=INK_PRIMARY)
    ax.set_xlabel("")
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=9)
    ax.tick_params(axis="x", colors=INK_PRIMARY, labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=True)

    # No in-image title/caption -- poster space is tight and this context
    # (n=450/condition, seeds 0-14, clean_2M checkpoint) lives in the
    # POSTER_DRAFT.md caption instead, so bbox_inches="tight" crops right
    # to the plot instead of leaving room for 3 lines of header text.
    _savefig(fig, "fig1_success_by_condition.png")


# ---------------------------------------------------------------------------
# Fig 2 -- recovery-response latency (detection -> full response), dumbbell
# ---------------------------------------------------------------------------

def build_fig2(latency_df: pd.DataFrame) -> None:
    plot_df = latency_df.sort_values("detection_delay_mean", ascending=True).copy()
    plot_df["label"] = plot_df["condition"].map(lambda c: _cond_label(c).replace("\n", " "))
    order = plot_df["label"].tolist()
    # Numeric y positions, not the string "label" column, are plotted
    # directly. Mixing manual set_yticklabels() with later string-keyed
    # ax.plot()/scatterplot() calls lets matplotlib's categorical-unit
    # converter build its own label->position table from first-appearance
    # order across those calls -- and the connector-line loop below only
    # touches the 4 conditions with a non-null response delay, so it was
    # silently registering those 4 labels at the wrong positions before the
    # full scatterplot ever ran, scrambling which dot belongs to which row.
    plot_df["y"] = range(len(plot_df))
    pos = dict(zip(plot_df["label"], plot_df["y"]))

    long_rows = []
    for _, row in plot_df.iterrows():
        long_rows.append({"y": row["y"], "stage": "Detection", "steps": row["detection_delay_mean"]})
        if pd.notna(row["response_delay_mean"]):
            long_rows.append({"y": row["y"], "stage": "Full response", "steps": row["response_delay_mean"]})
    long_df = pd.DataFrame(long_rows)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))

    # connector lines first, so dots draw on top
    for _, row in plot_df.iterrows():
        if pd.notna(row["response_delay_mean"]):
            ax.plot(
                [row["detection_delay_mean"], row["response_delay_mean"]],
                [row["y"], row["y"]],
                color=INK_MUTED, linewidth=1.4, alpha=0.55, zorder=1,
            )

    palette = {"Detection": HX6_COLOR, "Full response": "#3d3d3d"}
    sns.scatterplot(
        data=long_df, y="y", x="steps", hue="stage", style="stage",
        palette=palette, s=110, edgecolor=INK_PRIMARY, linewidth=0.7,
        ax=ax, zorder=2,
    )

    for _, row in plot_df.iterrows():
        note = "" if pd.notna(row["response_delay_mean"]) else "  (no full ramp within episode)"
        tr = row["trigger_rate"] * 100
        x_end = row["response_delay_mean"] if pd.notna(row["response_delay_mean"]) else row["detection_delay_mean"]
        ax.text(
            x_end + 2.5, row["y"], f"trigger {tr:.0f}%{note}",
            va="center", fontsize=7.3, color=INK_MUTED,
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(len(order) - 0.5, -0.5)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Steps into episode", fontsize=10.5, color=INK_PRIMARY)
    ax.set_ylabel("")
    ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=9)
    ax.tick_params(axis="y", colors=INK_PRIMARY, labelsize=10)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True, bottom=False)
    # Placed outside the axes (not "lower right" etc.) because dense
    # per-row annotation text runs the full width of the plot at every
    # vertical position -- any inside corner collides with some row's label.
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)

    # No in-image title -- see the note in build_fig1; this figure's caption
    # (mean steps, triggering episodes only, 150-step ramp cutoff) lives in
    # POSTER_DRAFT.md instead.
    _savefig(fig, "fig2_recovery_latency.png")


# ---------------------------------------------------------------------------
# Compact evaluation table
# ---------------------------------------------------------------------------

def build_table(success_df, latency_df, safety_df) -> pd.DataFrame:
    df = success_df.merge(latency_df, on="condition", how="left")
    df = df.merge(safety_df, on="condition", how="left")
    df["condition"] = pd.Categorical(df["condition"], categories=ALL_CONDITIONS, ordered=True)
    df = df.sort_values("condition")

    out = pd.DataFrame({
        "condition": df["condition"],
        "trigger_rate_pct": (df["trigger_rate"] * 100).round(1),
        "success_rate_pct": (df["success_rate"] * 100).round(1),
        "detection_delay_steps": df["detection_delay_mean"].round(1),
        "response_delay_steps": df["response_delay_mean"].round(1),
        "safety_violation_pct": (df["safety_violation_rate"] * 100).round(2),
    })
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/hx6_solo_evaluation_table.csv", index=False)  # real NaN, for programmatic use
    with open("results/hx6_solo_evaluation_table.md", "w") as f:
        f.write(f"# {HX6_LABEL} — compact evaluation table\n\n")
        f.write("n=450 episodes/condition, seeds 0-14, clean_2M checkpoint. "
                "Detection/response delay: mean steps over triggering episodes; "
                "response delay blank where the blend never reaches full strength "
                "within the 150-step episode.\n\n")
        f.write(_df_to_markdown(out.fillna("—")))
        f.write("\n")
    print("wrote results/hx6_solo_evaluation_table.csv")
    print("wrote results/hx6_solo_evaluation_table.md")
    return out


# ---------------------------------------------------------------------------
# Fig 3 -- compact evaluation table, rendered as an image
# ---------------------------------------------------------------------------
# Seaborn has no native table mark, so this is a matplotlib table.table()
# styled with the same seaborn-set theme/tokens (HX6_COLOR, INK_*, gridline
# tint) as fig1/fig2, kept in this script rather than a plain markdown table
# so all three results panels share one visual language.

# Short condition labels for this figure only -- _cond_label's normal
# 2-line wrapped names (e.g. "Goal Spoof\n(mid-ep)") run too tall once
# flattened to one line for a table row ("Goal Spoof (mid-ep)", 20 chars).
_FIG3_LABELS = {
    "sensor_dropout":            "Sensor Dropout",
    "sensor_bias":                "Sensor Bias",
    "action_clipping":            "Action Clipping",
    "action_delay":                "Action Delay",
    "action_reversal":            "Action Reversal",
    "goal_spoof_immediate":      "Goal Spoof–Imm.",
    "goal_spoof_midep":          "Goal Spoof–Mid-Ep.",
    "object_pose_spoof":          "Object Pose Spoof",
    "grip_state_falsification":  "Grip Falsify",
    "contact_dropout":            "Contact Dropout",
}
# The four conditions where TAIRO-HX's own numbers (Fig. 1/2, the
# goal-spoofing writeup) are actually doing narrative work -- highlighted;
# everything else is present for completeness but muted so the table reads
# as a reference strip, not a second competing headline chart.
_FIG3_HIGHLIGHT = {"action_clipping", "action_delay", "goal_spoof_immediate", "goal_spoof_midep"}


def build_fig3_table(out: pd.DataFrame) -> None:
    # Clean is dropped -- its 100.0% clean-task performance is already
    # called out in the "Final quantitative findings" bullets above this
    # table, so repeating it here just adds a row for no new information.
    df = out[out["condition"].astype(str) != "clean"].copy()
    df["condition"] = df["condition"].astype(str)
    df["Condition"] = df["condition"].map(_FIG3_LABELS)

    def fmt_pct0(v):
        return "—" if pd.isna(v) else f"{v:.0f}%"

    def fmt_pct1(v):
        return "—" if pd.isna(v) else f"{v:.1f}%"

    def fmt_steps0(v):
        return "—" if pd.isna(v) else f"{v:.0f}"

    # Success leads (it's the metric people care about first), trigger
    # rate follows -- reversed from the previous trigger-before-success
    # order.
    col_labels = ["Condition", "Success", "Trigger", "Detect", "Full Response", "Safety Viol."]
    rows = [
        [r["Condition"], fmt_pct0(r["success_rate_pct"]), fmt_pct0(r["trigger_rate_pct"]),
         fmt_steps0(r["detection_delay_steps"]), fmt_steps0(r["response_delay_steps"]),
         fmt_pct1(r["safety_violation_pct"])]
        for _, r in df.iterrows()
    ]
    conditions = df["condition"].tolist()

    n_rows = len(rows)
    col_widths = [0.22, 0.13, 0.13, 0.12, 0.22, 0.18]
    # Poster reference-strip sizing: same full width as before. font size,
    # PAD, and table.scale together (not figsize -- tight-bbox crops to the
    # table's actual rendered extent) are tuned for a ~35-40% shorter table
    # than the previous pass while keeping all 10 rows legible.
    fig, ax = plt.subplots(figsize=(11.4, 0.30 * n_rows + 0.25))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                      colWidths=col_widths, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.3)

    HIGHLIGHT_BG = "#fdf1d6"   # light tint of HX6_COLOR
    MUTED_BG = "#f4f4f2"       # light gray -- recedes rows that aren't the story
    GRID_COLOR = "#ebebe6"     # thin/subtle, one step lighter than the previous pass
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.PAD = 0.028  # tighter than the previous pass's default padding, not as tight as a first attempt overshot
        if r == 0:
            cell.set_facecolor(HX6_COLOR)
            cell.set_edgecolor(HX6_COLOR)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(cell.get_height() * 0.85)  # header slightly shorter than body rows
            if c == 0:
                cell.get_text().set_ha("left")
            continue
        highlighted = conditions[r - 1] in _FIG3_HIGHLIGHT
        cell.set_facecolor(HIGHLIGHT_BG if highlighted else MUTED_BG)
        cell.set_edgecolor(GRID_COLOR)
        text_color = INK_PRIMARY if highlighted else INK_MUTED
        cell.get_text().set_color(text_color)
        if c == 0:
            cell.get_text().set_fontweight("bold" if highlighted else "normal")
            cell.get_text().set_ha("left")

    # No in-image title -- see the note in build_fig1; this table's context
    # (n=450/condition, seeds 0-14, clean_2M checkpoint) lives in
    # POSTER_DRAFT.md's caption instead. pad_inches trimmed to the minimum
    # that still keeps the header fill from clipping at the figure edge.
    _savefig(fig, "fig3_evaluation_table.png", pad_inches=0.03)


def main():
    success_df = load_success_by_condition()
    latency_df = load_latency_by_condition()
    safety_df = load_safety_by_condition()

    build_fig1(success_df)
    build_fig2(latency_df)
    table = build_table(success_df, latency_df, safety_df)
    build_fig3_table(table)

    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
