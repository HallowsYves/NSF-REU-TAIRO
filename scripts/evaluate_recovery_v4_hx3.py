"""
Evaluate Recovery v4-HX3 (relocalization_expert re-gated on Level 4's
perception_state signal) against v4 and v4_hx2, plus a do-no-harm check
against plain sac_her across all 11 conditions.

Bounded, one-pass evaluation per the 2026-07-21 plan: hx3 is the only newly
collected method (results/data_recovery_v4_hx3/, n=450, seeds 0-14, all 11
conditions); sac_her/v4/v4_hx2 baselines are reused from existing sources
already collected at the same power level earlier this session. No new
statistical code -- reuses compare_two_methods/bh_adjust from
scripts/audit_recovery_do_no_harm.py.

Output: results/recovery_v4_hx3_evaluation.csv
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

DATA_SOURCES = _BASE_DATA_SOURCES + ["results/data_recovery_v4_hx3"]


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
    print("\n[eval] Seed coverage per method x condition (hx3 row should show 15 everywhere):")
    print(coverage.to_string())

    # -- hx3 vs v4 and hx3 vs hx2, full 11-condition grid --------------------
    rows = []
    for baseline in ["sac_her_recovery_v4", "sac_her_recovery_v4_hx2"]:
        for condition in ALL_CONDITIONS:
            row = compare_two_methods(combined, "sac_her_recovery_v4_hx3", baseline, condition)
            if row is not None:
                rows.append(row)
    vs_df = pd.DataFrame(rows)
    vs_df["p_value_bh"] = bh_adjust(vs_df["p_value"].values)
    vs_df["significant_bh"] = vs_df["p_value_bh"] < 0.05

    print("\n[eval] hx3 vs v4 on object_pose_spoof (the targeted fix):")
    print(vs_df[(vs_df.baseline_method == "sac_her_recovery_v4") &
                (vs_df.condition == "object_pose_spoof")]
          .to_string(index=False))

    print("\n[eval] hx3 vs hx2 on object_pose_spoof (does it beat the adopted method?):")
    print(vs_df[(vs_df.baseline_method == "sac_her_recovery_v4_hx2") &
                (vs_df.condition == "object_pose_spoof")]
          .to_string(index=False))

    print("\n[eval] hx3 vs hx2 on grip_state_falsification (must confirm the existing win survives):")
    print(vs_df[(vs_df.baseline_method == "sac_her_recovery_v4_hx2") &
                (vs_df.condition == "grip_state_falsification")]
          .to_string(index=False))

    print("\n[eval] All hx3-vs-v4 / hx3-vs-hx2 comparisons reaching BH-corrected significance:")
    sig = vs_df[vs_df.significant_bh]
    print(sig.to_string(index=False) if len(sig) else "        (none)")

    # -- hx3 vs sac_her, full do-no-harm check across all 11 conditions ------
    dnh_rows = []
    for condition in ALL_CONDITIONS:
        row = compare_two_methods(combined, "sac_her_recovery_v4_hx3", "sac_her", condition)
        if row is not None:
            dnh_rows.append(row)
    dnh_df = pd.DataFrame(dnh_rows)
    dnh_df["p_value_bh"] = bh_adjust(dnh_df["p_value"].values)
    dnh_df["significant_bh"] = dnh_df["p_value_bh"] < 0.05

    print("\n[eval] hx3 vs sac_her, do-no-harm check across all 11 conditions:")
    print(dnh_df[["condition", "baseline_success_rate", "recovery_success_rate",
                  "delta", "p_value", "p_value_bh", "significant_bh"]].to_string(index=False))

    harm = dnh_df[(dnh_df.delta < 0) & dnh_df.significant_bh]
    print(f"\n[eval] {len(harm)} conditions show significant harm vs sac_her (BH-adjusted p<0.05):")
    print(harm.to_string(index=False) if len(harm) else "        (none)")

    out_path = "results/recovery_v4_hx3_evaluation.csv"
    combined_out = pd.concat([vs_df.assign(comparison="vs_v4_or_hx2"),
                               dnh_df.assign(comparison="vs_sac_her")], ignore_index=True)
    combined_out.to_csv(out_path, index=False)
    print(f"\n[eval] Wrote {len(combined_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
