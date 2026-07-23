"""
Evaluate Recovery v4-HX6 (Level-4-gated fast-attack trigger EMA, see
recovery/recovery_v4_hx6.py) against v4, v4_hx2, sac_her, AND v2/v3 --
all now available at matched full power (n=450, seeds 0-14, all 11
conditions) as of this session's v2/v3 backfill
(results/data_recovery_v4_v2v3_backfill/), unlike hx4/hx5's evaluation
scripts which had to treat v3 as a separate, mismatched-n aside.

Two questions, both answered at full BH-corrected power:
1. Does hx6 close any of the goal-spoof gap vs. v3 that motivated the
   whole hx4/hx5/hx6 trigger-speed investigation?
2. Does hx6 close any of the grip_state_falsification gap where v2/v3
   significantly outperform the adopted v4-HX2 (this session's finding,
   scripts/build_final_hx_comparison.py) -- hx6 was specifically designed
   so the fast-attack multiplier CANNOT fire on that condition (Level-4
   gated to perception_state/goal_manipulation only), so the honest
   expectation is "no change there" -- confirmed, not assumed.

No new statistical code -- reuses compare_two_methods/bh_adjust from
scripts/audit_recovery_do_no_harm.py.

Output: results/recovery_v4_hx6_evaluation.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import ALL_CONDITIONS
from scripts.audit_recovery_do_no_harm import (
    DATA_SOURCES as _BASE_DATA_SOURCES,
    EP_FILENAME,
    compare_two_methods,
    bh_adjust,
)

DATA_SOURCES = _BASE_DATA_SOURCES + [
    "results/data_recovery_v4_hx6",
    "results/data_recovery_v4_v2v3_backfill",
]
GOAL_SPOOF_CONDITIONS = ["goal_spoof_immediate", "goal_spoof_midep"]
TARGET_METHOD = "sac_her_recovery_v4_hx6"


def load_all_episodes() -> pd.DataFrame:
    frames = []
    for src in DATA_SOURCES:
        path = os.path.join(src, EP_FILENAME)
        if not os.path.exists(path):
            print(f"[eval] WARNING: missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        df["_source"] = src
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["method", "condition", "seed"]).reset_index(drop=True)
    combined["episode_in_seed"] = combined.groupby(["method", "condition", "seed"]).cumcount()
    dup_key = ["method", "condition", "seed", "episode_in_seed"]
    n_dupes = combined.duplicated(subset=dup_key).sum()
    if n_dupes:
        raise ValueError(
            f"[eval] {n_dupes} duplicate (method, condition, seed, episode_in_seed) "
            "rows found across data sources -- investigate before trusting results."
        )
    return combined


def main():
    combined = load_all_episodes()
    print(f"[eval] Loaded {len(combined)} episodes from {len(DATA_SOURCES)} sources.")

    coverage = combined.groupby(["method", "condition"])["seed"].nunique().unstack(fill_value=0)
    print("\n[eval] Seed coverage per method x condition (hx6/v2/v3 rows should show 15 everywhere):")
    print(coverage.to_string())

    # -- hx6 vs v4/hx2/v2/v3, full 11-condition grid (n=450, paired) ---------
    rows = []
    for baseline in ["sac_her_recovery_v4", "sac_her_recovery_v4_hx2",
                      "sac_her_recovery_v2", "sac_her_recovery_v3"]:
        for condition in ALL_CONDITIONS:
            row = compare_two_methods(combined, TARGET_METHOD, baseline, condition)
            if row is not None:
                rows.append(row)
    vs_df = pd.DataFrame(rows)
    vs_df["p_value_bh"] = bh_adjust(vs_df["p_value"].values)
    vs_df["significant_bh"] = vs_df["p_value_bh"] < 0.05

    print("\n[eval] hx6 vs v4/hx2/v2/v3, goal-spoof conditions (the original targeted fix):")
    print(vs_df[vs_df.condition.isin(GOAL_SPOOF_CONDITIONS)].to_string(index=False))

    print("\n[eval] hx6 vs v4/hx2/v2/v3, grip_state_falsification "
          "(expected: no change vs hx2, since hx6 is Level-4-gated OFF this condition):")
    print(vs_df[vs_df.condition == "grip_state_falsification"].to_string(index=False))

    print("\n[eval] All hx6-vs-{v4,hx2,v2,v3} comparisons reaching BH-corrected significance:")
    sig = vs_df[vs_df.significant_bh]
    print(sig.to_string(index=False) if len(sig) else "        (none)")

    # -- hx6 vs sac_her, full do-no-harm check across all 11 conditions ------
    dnh_rows = []
    for condition in ALL_CONDITIONS:
        row = compare_two_methods(combined, TARGET_METHOD, "sac_her", condition)
        if row is not None:
            dnh_rows.append(row)
    dnh_df = pd.DataFrame(dnh_rows)
    dnh_df["p_value_bh"] = bh_adjust(dnh_df["p_value"].values)
    dnh_df["significant_bh"] = dnh_df["p_value_bh"] < 0.05

    print("\n[eval] hx6 vs sac_her, do-no-harm check across all 11 conditions:")
    print(dnh_df[["condition", "baseline_success_rate", "recovery_success_rate",
                  "delta", "p_value", "p_value_bh", "significant_bh"]].to_string(index=False))

    harm = dnh_df[(dnh_df.delta < 0) & dnh_df.significant_bh]
    print(f"\n[eval] {len(harm)} conditions show significant harm vs sac_her (BH-adjusted p<0.05):")
    print(harm.to_string(index=False) if len(harm) else "        (none)")

    benefit = dnh_df[(dnh_df.delta > 0) & dnh_df.significant_bh]
    print(f"\n[eval] {len(benefit)} conditions show significant BENEFIT vs sac_her (BH-adjusted p<0.05):")
    print(benefit.to_string(index=False) if len(benefit) else "        (none)")

    out_path = "results/recovery_v4_hx6_evaluation.csv"
    combined_out = pd.concat([
        vs_df.assign(comparison="vs_v4_hx2_v2_v3"),
        dnh_df.assign(comparison="vs_sac_her"),
    ], ignore_index=True)
    combined_out.to_csv(out_path, index=False)
    print(f"\n[eval] Wrote {len(combined_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
