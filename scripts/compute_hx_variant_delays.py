"""
Per-condition detection/response-delay + trigger-rate for the three HX
variants scripts/build_final_hx_comparison.py never computed this for --
HX, HX2, HX3, HX4, HX5 (only v4 and the adopted HX6 got a full pass, saved
to results/final_hx_comparison_delays.csv).

Reproduces that script's exact step-level reduction (same onset_step /
RESPONSE_W_THRESH=0.5 "full response" definition, same exclusion of the
`clean` condition -- sac_her/no attack has no meaningful "attack response
delay", not a 0) against each variant's own raw step_logs CSV, since no
precomputed delay table exists for these five variants anywhere in the repo.

Sample sizes are NOT uniform across variants -- HX/HX2 each have only
n=150/condition (their original evaluation run, before the later variants
were re-run at full n=450 power); HX3/HX4/HX5 have n=450/condition, matching
v4/HX6. Kept as-is rather than subsampling HX3-5 down to 150 or pretending
HX/HX2 have more data than they do -- see the `n_episodes` column in the
output for the honest per-row denominator.

Output: results/hx_variants_hx_hx2_hx3_hx4_hx5_delays.csv
    (same schema as results/final_hx_comparison_delays.csv, so the figure
    script can concat the two directly)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import ALL_CONDITIONS
from evaluation.attack_dispatch import GOAL_SPOOF_MIDEP_STEP

RESPONSE_W_THRESH = 0.5
STEP_USECOLS = ["method", "condition", "seed", "episode_idx", "timestep",
                 "recovery_triggered", "recovery_v4_weight"]

# (variant label, step_logs path, method filter -- None means the file
# contains exactly one method already)
SOURCES = [
    ("HX",  "results/data_recovery_v4_hx/step_logs_sac_her_pickandplace_clean_2M.csv", None),
    ("HX2", "results/data_recovery_v4_hx2/step_logs_sac_her_pickandplace_clean_2M.csv", None),
    ("HX3", "results/data_recovery_v4_hx3/step_logs_sac_her_pickandplace_clean_2M.csv", "sac_her_recovery_v4_hx3"),
    ("HX4", "results/data_recovery_v4_hx4/step_logs_sac_her_pickandplace_clean_2M.csv", "sac_her_recovery_v4_hx4"),
    ("HX5", "results/data_recovery_v4_hx5/step_logs_sac_her_pickandplace_clean_2M.csv", "sac_her_recovery_v4_hx5"),
]

OUT_CSV = "results/hx_variants_hx_hx2_hx3_hx4_hx5_delays.csv"


def onset_step(condition: str) -> int:
    return GOAL_SPOOF_MIDEP_STEP if condition == "goal_spoof_midep" else 0


def reduce_step_log(path: str, method_filter: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=STEP_USECOLS)
    if method_filter is not None:
        df = df[df["method"] == method_filter]
    df = df[df["condition"] != "clean"]  # no attack -> no meaningful response delay
    df = df.sort_values(["method", "condition", "seed", "episode_idx", "timestep"])

    rows = []
    for (method, condition, seed, ep_idx), g in df.groupby(
        ["method", "condition", "seed", "episode_idx"], sort=False
    ):
        onset = onset_step(condition)
        g = g[g["timestep"] >= onset]
        triggered = g["recovery_triggered"].fillna(0).astype(int).values
        timesteps = g["timestep"].values

        trig_idx = np.flatnonzero(triggered == 1)
        if len(trig_idx) == 0:
            detection_delay = np.nan
            response_delay = np.nan
        else:
            first_trigger_t = timesteps[trig_idx[0]]
            detection_delay = float(first_trigger_t - onset)
            w = g["recovery_v4_weight"].fillna(0.0).values
            post = np.flatnonzero((w >= RESPONSE_W_THRESH) & (timesteps >= first_trigger_t))
            response_delay = float(timesteps[post[0]] - first_trigger_t) if len(post) else np.nan

        rows.append({
            "method": method, "condition": condition,
            "detection_delay": detection_delay, "response_delay": response_delay,
        })
    return pd.DataFrame(rows)


def main() -> None:
    delay_rows = []
    for variant, path, method_filter in SOURCES:
        print(f"[hx-delays] reducing {variant}: {path}")
        ep_df = reduce_step_log(path, method_filter)
        method_name = ep_df["method"].iloc[0]
        for condition in [c for c in ALL_CONDITIONS if c != "clean"]:
            sub = ep_df[ep_df["condition"] == condition]
            n_episodes = len(sub)
            det = sub["detection_delay"].dropna()
            resp = sub["response_delay"].dropna()
            delay_rows.append({
                "method": method_name, "condition": condition, "n_episodes": n_episodes,
                "trigger_rate": len(det) / n_episodes if n_episodes else np.nan,
                "detection_delay_mean": det.mean() if len(det) else np.nan,
                "detection_delay_median": det.median() if len(det) else np.nan,
                "response_delay_mean": resp.mean() if len(resp) else np.nan,
                "response_delay_median": resp.median() if len(resp) else np.nan,
            })

    out = pd.DataFrame(delay_rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[hx-delays] wrote {len(out)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
