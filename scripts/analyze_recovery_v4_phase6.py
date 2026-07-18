"""
Recovery v4 Tier 1 (CCAR) -- Phase 6 evaluation analysis (clean_2M).

Reads the full 4-method multi-seed sweep (sac_her, recovery_v2, recovery_v3,
recovery_v4) on the clean_2M PickAndPlace checkpoint and reports:

  1. Clean-condition success rate per method -- the specific check that v4's
     residual clean-episode blending does NOT degrade task success relative
     to no-recovery / v2 / v3.
  2. Per-condition success rate for all 4 methods (aggregate over 5 seeds x
     30 episodes = 150 episodes per cell -- multi-seed aggregate, not a
     single-run point estimate, per the RNG-non-reproducibility caveat in
     findings.md).
  3. Composite trustworthiness score (C1-C5, the paper's Eq. 2 weighted
     composite) per method, computed via the UNMODIFIED evaluation/metrics.py
     (summarize_results + add_trustworthiness_scores).

Why this script exists rather than scripts/build_benchmark_table.py: that
script only forwards the B1 sac_her no-recovery baseline to layers {B2, B3}
for the C5 recovery_score formula, NOT to B4 (recovery_v4). Running it
unmodified would give v4's C5 recovery_score = its full success rate instead
of its improvement over no-recovery. This script passes the B1 baseline
explicitly for B4 as well. It imports metrics.py read-only and does not
modify it or build_benchmark_table.py (both are out of scope to edit). The
B4 gap in build_benchmark_table.py is flagged for a separate fix if v4
becomes paper-facing.

Terminology note: the prompt referenced "the standalone Eq. 2 script." The
standalone scripts/compute_trustworthiness_s1s4.py computes Eq. 1 (S1-S4)
only and explicitly excludes S5/T (no recovery data existed when it was
written). The full composite T (Eq. 2, with C5 recovery) is in metrics.py,
which is what a recovery-method comparison actually needs -- used here.

Nothing in this script draws a conclusion about whether v4 beats v2/v3; it
reports numbers for mentor review.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import ALL_CONDITIONS
from evaluation.metrics import summarize_results, add_trustworthiness_scores

METHOD_ORDER = ["sac_her", "sac_her_recovery_v2", "sac_her_recovery_v3", "sac_her_recovery_v4"]
# Layers that are recovery methods and therefore need the B1 sac_her baseline
# forwarded for the C5 recovery_score formula. B4 is the addition vs
# build_benchmark_table.py.
RECOVERY_LAYERS = {"B2", "B3", "B4"}

_parser = argparse.ArgumentParser()
_parser.add_argument("--input-file", type=str,
                     default="results/data_recovery_v4/episode_results_sac_her_pickandplace_clean_2M.csv")
_args = _parser.parse_args()

print(f"[phase6] Loading {_args.input_file}")
df = pd.read_csv(_args.input_file)
print(f"[phase6] {len(df)} episodes  methods={sorted(df['method'].unique())}  "
      f"seeds={sorted(df['seed'].unique())}  conditions={df['condition'].nunique()}\n")

# ---------------------------------------------------------------------------
# 1. Clean-condition success rate per method (the degradation check)
# ---------------------------------------------------------------------------
print("=" * 78)
print("1. CLEAN-CONDITION success rate per method (v4 must not degrade vs baselines)")
print("=" * 78)
clean = df[df["condition"] == "clean"]
for m in METHOD_ORDER:
    sub = clean[clean["method"] == m]
    if len(sub) == 0:
        continue
    # per-seed success rate then mean +/- std across seeds
    per_seed = sub.groupby("seed")["success"].mean()
    print(f"  {m:<24} success = {per_seed.mean():.4f} +/- {per_seed.std():.4f}  "
          f"(n={len(sub)} episodes, {len(per_seed)} seeds)")

# ---------------------------------------------------------------------------
# 2. Per-condition success rate, all methods (multi-seed aggregate)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. PER-CONDITION success rate (mean over 5 seeds x 30 eps = 150 eps/cell)")
print("=" * 78)
hdr = f"  {'condition':<26}" + "".join(f"{m.replace('sac_her_recovery_','v').replace('sac_her','base'):>10}" for m in METHOD_ORDER)
print(hdr)
for cond in ALL_CONDITIONS:
    row = f"  {cond:<26}"
    for m in METHOD_ORDER:
        sub = df[(df["condition"] == cond) & (df["method"] == m)]
        val = sub["success"].mean() if len(sub) else float("nan")
        row += f"{val:>10.3f}"
    print(row)

# ---------------------------------------------------------------------------
# 3. Composite T-score (Eq. 2, C1-C5) per method, via unmodified metrics.py
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. COMPOSITE TRUSTWORTHINESS (Eq. 2 weighted C1-C5), per method")
print("   B1 sac_her baseline forwarded to B2/B3/B4 for the C5 recovery_score.")
print("=" * 78)

b1_df = df[df["benchmark_layer"] == "B1"]
b1_baseline = summarize_results(b1_df) if not b1_df.empty else None

layer_summaries = []
for layer in sorted(df["benchmark_layer"].unique()):
    layer_df = df[df["benchmark_layer"] == layer]
    if layer_df.empty:
        continue
    summ = summarize_results(layer_df)
    baseline = b1_baseline if layer in RECOVERY_LAYERS else None
    summ = add_trustworthiness_scores(summ, baseline_summary=baseline)
    summ.insert(0, "benchmark_layer", layer)
    layer_summaries.append(summ)

summary_df = pd.concat(layer_summaries, ignore_index=True)

# Aggregate composite T per method as the mean across all conditions (the
# convention used elsewhere: one composite per method averaged over conditions).
print(f"  {'method':<24}{'T_weighted':>14}{'T_equal':>12}{'C1_reliab':>12}"
      f"{'C3_cyber':>11}{'C4_safety':>11}{'C5_recov':>11}")
for m in METHOD_ORDER:
    sub = summary_df[summary_df["method"] == m]
    if len(sub) == 0:
        continue
    print(f"  {m:<24}"
          f"{sub['trustworthiness_score_weighted'].mean():>14.4f}"
          f"{sub['trustworthiness_score_equal'].mean():>12.4f}"
          f"{sub['reliability_score'].mean():>12.4f}"
          f"{sub['cyber_resilience_score'].mean():>11.4f}"
          f"{sub['safety_score'].mean():>11.4f}"
          f"{sub['recovery_score'].mean():>11.4f}")

out_summary = _args.input_file.replace("episode_results_", "phase6_summary_")
summary_df.to_csv(out_summary, index=False)
print(f"\n[phase6] Full per-(layer,method,condition) summary -> {out_summary}")
print("[phase6] Numbers are for mentor review; no v4-vs-v2/v3 conclusion drawn here.")
