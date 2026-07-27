"""
Latency comparison across all HX variants, built with seaborn (2026-07-24).

Companion to scripts/build_hx_all_variants_seaborn_figure.py (success/safety
across HX..HX6) -- this one covers the timing side: how quickly each variant
detects an attack (trigger delay) and how quickly its blend weight reaches
"full/meaningful" recovery authority (w >= 0.5, response delay), plus how
often it triggers at all.

Same ordinal-progression treatment as the success/safety figure: HX..HX5 are
an oldest->newest research iteration, drawn as a light->dark one-hue ramp;
HX6 (adopted) keeps the established yellow. v4 (the common CCAR baseline
every HX variant extends) is deliberately not drawn as its own series here
(confirmed via sign-off, 2026-07-24, matching the same scope decision on the
companion success/safety figure) -- this figure is about the HX iteration
sequence's own timing, not a v4-vs-HX comparison. Same validated palette
("#86b6ef,#5598e7,#2a78d6,#1c5cab,#104281" ordinal + "#eda100" endpoint) --
see build_hx_all_variants_seaborn_figure.py's docstring for the validator
invocations.

Data provenance (no new evaluation beyond scripts/compute_hx_variant_delays.py):
- HX6: results/final_hx_comparison_delays.csv (already computed by
  scripts/build_final_hx_comparison.py, n=450/condition seeds 0-14).
- HX, HX2, HX3, HX4, HX5: results/hx_variants_hx_hx2_hx3_hx4_hx5_delays.csv
  (scripts/compute_hx_variant_delays.py -- same exact reduction algorithm:
  same onset_step/RESPONSE_W_THRESH=0.5 definition, same exclusion of
  `clean`, reused rather than reimplemented). HX/HX2 are n=150/condition
  (their original evaluation power); HX3/HX4/HX5 are n=450/condition. This
  is a real, flagged sample-size asymmetry, not an oversight -- see that
  script's docstring.

All three panels aggregate the 10 non-clean per-condition means with an
UNWEIGHTED mean-of-condition-means (matching how
results/final_hx_comparison_summary_table.csv's own detection/response-delay
columns were built -- verified: (mean of hx6_solo_evaluation_table.md's 10
per-condition detection_delay values) == 20.03 == that table's hx6 row).
NOT an episode-weighted pool -- a condition that rarely triggers (e.g.
action_clipping, ~5%) still gets equal say in the average, same as a
condition that always triggers.

Response delay is only defined over episodes that ever reach the w>=0.5
threshold -- for the L4-downweight variants (HX2/HX3/HX4) roughly half of
the 10 attacked conditions never cross it at all (the downweight keeps the
blend below "full" by design on confidently-flagged action_actuation
episodes), so each response-delay bar is annotated with how many of the 10
conditions it's actually averaged over -- comparing a mean-of-4 to a
mean-of-10 at face value would be misleading.

Output:
    results/figures/final_hx_comparison/fig_hx_all_variants_latency.png (landscape, 3-across)
    results/figures/final_hx_comparison/fig_hx_all_variants_latency_vertical.png
        (portrait, 3 stacked -- same panels/data, for a narrow poster column)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = "results/figures/final_hx_comparison"
OFFICIAL_DELAYS_CSV = "results/final_hx_comparison_delays.csv"
HX_FAMILY_DELAYS_CSV = "results/hx_variants_hx_hx2_hx3_hx4_hx5_delays.csv"

VARIANT_ORDER = ["HX", "HX2", "HX3", "HX4", "HX5", "HX6"]
VARIANT_LABELS = {
    "HX": "HX", "HX2": "HX2", "HX3": "HX3",
    "HX4": "HX4", "HX5": "HX5", "HX6": "HX6",
}
METHOD_NAME = {
    "HX": "sac_her_recovery_v4_hx",
    "HX2": "sac_her_recovery_v4_hx2", "HX3": "sac_her_recovery_v4_hx3",
    "HX4": "sac_her_recovery_v4_hx4", "HX5": "sac_her_recovery_v4_hx5",
    "HX6": "sac_her_recovery_v4_hx6",
}
VARIANT_COLORS = {
    "HX":  "#86b6ef",
    "HX2": "#5598e7",
    "HX3": "#2a78d6",
    "HX4": "#1c5cab",
    "HX5": "#104281",
    "HX6": "#eda100",
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


def _savefig(fig, name, pad_inches=0.1):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"wrote {path}")


def load_per_condition() -> pd.DataFrame:
    official = pd.read_csv(OFFICIAL_DELAYS_CSV)
    official = official[official["method"] == METHOD_NAME["HX6"]]
    family = pd.read_csv(HX_FAMILY_DELAYS_CSV)
    return pd.concat([official, family], ignore_index=True)


def aggregate(per_cond: pd.DataFrame) -> pd.DataFrame:
    name_to_variant = {v: k for k, v in METHOD_NAME.items()}
    per_cond = per_cond.copy()
    per_cond["variant"] = per_cond["method"].map(name_to_variant)

    rows = []
    for variant in VARIANT_ORDER:
        sub = per_cond[per_cond["variant"] == variant]
        n_conditions = len(sub)
        n_episodes = int(sub["n_episodes"].iloc[0]) if "n_episodes" in sub.columns and len(sub) else None
        resp = sub["response_delay_mean"].dropna()
        rows.append({
            "variant": variant,
            "detection_delay": sub["detection_delay_mean"].mean(),
            "trigger_rate": sub["trigger_rate"].mean(),
            "response_delay": resp.mean() if len(resp) else float("nan"),
            "response_n_conditions": len(resp),
            "n_conditions": n_conditions,
            "n_episodes_per_condition": n_episodes,
        })
    out = pd.DataFrame(rows)
    out["variant"] = pd.Categorical(out["variant"], categories=VARIANT_ORDER, ordered=True)
    return out.sort_values("variant").reset_index(drop=True)


def _bar(ax, df, y_col, ylabel, ylim, pct=False, annotate_n=None):
    # Short codes only on the bars themselves -- the "+L4 down-weight" /
    # "+global fast trigger" style descriptions live in
    # fig_hx_all_variants_comparison.png (the companion success/safety
    # figure) and the surrounding write-up, not baked into every axis here.
    # Three side-by-side subplots leave far less width per bar than that
    # figure's single full-width panel, so the earlier two/three-line
    # per-bar labels collided between adjacent bars.
    order = VARIANT_ORDER
    labels = [VARIANT_LABELS[v] for v in order]
    colors = [VARIANT_COLORS[v] for v in order]
    sub = df.set_index("variant").loc[order].reset_index()

    bars = ax.bar(labels, sub[y_col], color=colors, edgecolor=INK_PRIMARY, linewidth=0.6, zorder=3)

    for bar, val, n in zip(bars, sub[y_col], sub[annotate_n] if annotate_n else [None] * len(sub)):
        if pd.isna(val):
            continue
        label = f"{val*100:.0f}%" if pct else f"{val:.1f}"
        if n is not None:
            label += f"\nn={n}/10"
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + ylim[1] * 0.015, label, ha="center", va="bottom",
                 fontsize=8.2, color=INK_PRIMARY, fontweight="bold", linespacing=1.5)

    ax.set_ylim(*ylim)
    if pct:
        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylabel(ylabel, fontsize=10, color=INK_PRIMARY)
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=9)
    ax.tick_params(axis="x", colors=INK_PRIMARY, labelsize=10.5)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=True)


def _draw_all_panels(ax_det, ax_resp, ax_trig, agg: pd.DataFrame) -> None:
    _bar(ax_det, agg, "detection_delay",
         "Detection delay (steps)\nattack onset -> first trigger", ylim=(0, 30))
    _bar(ax_resp, agg, "response_delay",
         "Response delay (steps)\nfirst trigger -> full authority (w>=0.5)", ylim=(0, 100),
         annotate_n="response_n_conditions")
    _bar(ax_trig, agg, "trigger_rate",
         "Trigger rate\n(share of attacked episodes)", ylim=(0, 1.08), pct=True)


def main() -> None:
    per_cond = load_per_condition()
    agg = aggregate(per_cond)
    print(agg.to_string())

    # Landscape, 3 panels side by side -- for full-width use (paper figure).
    fig, (ax_det, ax_resp, ax_trig) = plt.subplots(1, 3, figsize=(15.5, 5.6))
    _draw_all_panels(ax_det, ax_resp, ax_trig, agg)
    fig.tight_layout()
    _savefig(fig, "fig_hx_all_variants_latency.png")

    # Portrait, 3 panels stacked -- for a narrow poster column. Each panel
    # gets the figure's FULL width now (not a third of it), so the same
    # short-code x-labels have far more breathing room per bar than in the
    # landscape layout above.
    fig_v, (ax_det_v, ax_resp_v, ax_trig_v) = plt.subplots(3, 1, figsize=(6.0, 9.5))
    _draw_all_panels(ax_det_v, ax_resp_v, ax_trig_v, agg)
    fig_v.tight_layout(h_pad=1.6)
    _savefig(fig_v, "fig_hx_all_variants_latency_vertical.png")


if __name__ == "__main__":
    main()
