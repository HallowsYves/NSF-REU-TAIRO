"""
TAIRO-HX Level 5 -- recalibrate the two PROVISIONAL stop_safely proxy
thresholds (LEVEL5_LOW_CONFIDENCE_THRESH, LEVEL5_ABNORMAL_STREAK_THRESH)
against actual episode outcomes.

Motivated by scripts/evaluate_level5_decisions.py's finding: at the
original uncalibrated values (0.4, 2), stop_safely fires on 95.6% of
genuinely unrecoverable clean_2M episodes (good) but ALSO on 71.9% of
recoverable ones and 16.8% of no-problem ones (bad -- indistinguishable
from "sustained abnormality" rather than "truly unrecoverable"). Same
"calibrate from data, don't hand-pick" principle already used for the C4
safety threshold (multiplier x pooled-clean-max) and Recovery v4's
TriggerWeight midpoint/steepness (clean-rollout statistics).

Step 1: diagnose which of the two proxy rules (rule 2 = low-confidence
uncertainty, rule 3 = abnormal streak) is driving the false-fire rate.
Step 2: grid-search both thresholds, holding the SAME rule-2-OR-rule-3
structure (not changing the cascade logic, only its two constants), and
report the unrecoverable-hit-rate / recoverable-false-fire-rate /
no_problem-false-fire-rate tradeoff at each grid point.

This script does NOT decide the final threshold -- it produces the grid
for a human sign-off pick, same as this session's other judgment calls.
Once a choice is confirmed, update config.py's two constants and re-run
build_level5_labels.py + evaluate_level5_decisions.py to regenerate the
actual label/eval artifacts at the chosen values.

Input:  results/hierarchical_chain_predictions.csv (clean_2M rows)
        results/data_recovery_v4/episode_results_sac_her_pickandplace_clean_2M.csv
Output: printed diagnostic + grid tables only (no files written)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config

CHAIN_PATH = "results/hierarchical_chain_predictions.csv"
EP_PATH = "results/data_recovery_v4/episode_results_sac_her_pickandplace_clean_2M.csv"

# ── Load chain predictions, clean_2M only ────────────────────────────────
df = pd.read_csv(CHAIN_PATH)
df = df[df["model"] == "clean_2M"].copy()
df = df.sort_values(["seed", "episode_idx", "checkpoint_t"]).reset_index(drop=True)
df["level4_pred_class"] = df["level4_pred_class"].fillna("")

group_cols = ["seed", "episode_idx"]
is_abnormal = (df["level2_pred_class"] == "abnormal").astype(int)


def _prior_streak(s: pd.Series) -> pd.Series:
    streak = s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    streak = streak.where(s == 1, 0)
    return streak.shift(1, fill_value=0)


df["abnormal_streak_sofar"] = (
    is_abnormal.groupby([df[c] for c in group_cols]).transform(_prior_streak)
)

# ── Ground-truth partition (same construction as evaluate_level5_decisions.py) ─
ep = pd.read_csv(EP_PATH)
df["_local_idx"] = df["episode_idx"] - df.groupby(["seed", "condition"])["episode_idx"].transform("min")
ep["_local_idx"] = ep["episode_idx"] - ep.groupby(["method", "seed", "condition"])["episode_idx"].transform("min")

ep_wide = ep.pivot_table(index=["condition", "seed", "_local_idx"], columns="method",
                          values="success", aggfunc="first").reset_index()
ep_wide.columns.name = None
ep_wide = ep_wide.rename(columns={"sac_her": "success_sac_her", "sac_her_recovery_v4": "success_v4"})


def partition(row):
    if row["success_sac_her"] == 1:
        return "no_problem"
    if row["success_v4"] == 1:
        return "recoverable"
    return "unrecoverable"


ep_wide["partition"] = ep_wide.apply(partition, axis=1)
part_map = ep_wide.set_index(["condition", "seed", "_local_idx"])["partition"]
df["partition"] = df.set_index(["condition", "seed", "_local_idx"]).index.map(part_map)
assert df["partition"].isna().sum() == 0

# ── Rule-1 override (level3 == success) applies regardless of threshold ──
not_success_override = df["level3_pred_class"] != "success"

# ── Step 1: diagnose which rule drives current (0.4, 2) firings ─────────
rule2_fires = (
    not_success_override
    & (df["level2_pred_class"] == "abnormal")
    & (df["level3_confidence"] < 0.4)
    & (df["level4_confidence"] < 0.4)
)
rule3_fires = not_success_override & (df["abnormal_streak_sofar"] >= 2)

print("=" * 70)
print("STEP 1: which rule drives stop_safely at current (0.4, 2) thresholds?")
print("(row-level checkpoint counts, clean_2M)")
print("=" * 70)
diag = pd.DataFrame({
    "rule2_only": rule2_fires & ~rule3_fires,
    "rule3_only": rule3_fires & ~rule2_fires,
    "both": rule2_fires & rule3_fires,
    "partition": df["partition"],
})
print(diag.groupby("partition")[["rule2_only", "rule3_only", "both"]].sum().to_string())
print()


def episode_rate(streak_thresh, conf_thresh):
    r2 = (
        not_success_override
        & (df["level2_pred_class"] == "abnormal")
        & (df["level3_confidence"] < conf_thresh)
        & (df["level4_confidence"] < conf_thresh)
    )
    r3 = not_success_override & (df["abnormal_streak_sofar"] >= streak_thresh)
    fires = r2 | r3
    ep_fire = fires.groupby([df["condition"], df["seed"], df["_local_idx"]]).any()
    ep_fire = ep_fire.reset_index(name="ever_stop_safely")
    merged = ep_fire.merge(
        ep_wide[["condition", "seed", "_local_idx", "partition"]],
        on=["condition", "seed", "_local_idx"], how="left")
    return merged.groupby("partition")["ever_stop_safely"].mean()


print("=" * 70)
print("STEP 2: grid search (streak_thresh, conf_thresh) -> per-partition")
print("P(ever_stop_safely) -- want HIGH for unrecoverable, LOW for the rest")
print("=" * 70)
header = f"{'streak':>7} {'conf':>6} {'unrecov':>9} {'recov':>8} {'no_prob':>9}"
print(header)
for streak_thresh in [2, 3, 4, 5, 6, 8, 10]:
    for conf_thresh in [0.2, 0.3, 0.4]:
        rates = episode_rate(streak_thresh, conf_thresh)
        print(f"{streak_thresh:>7} {conf_thresh:>6.1f} "
              f"{rates.get('unrecoverable', float('nan')):>9.3f} "
              f"{rates.get('recoverable', float('nan')):>8.3f} "
              f"{rates.get('no_problem', float('nan')):>9.3f}")

print("\n[calibrate-level5] Grid printed. No files written -- pick a row, then "
      "update config.py's LEVEL5_LOW_CONFIDENCE_THRESH / "
      "LEVEL5_ABNORMAL_STREAK_THRESH and re-run build_level5_labels.py + "
      "evaluate_level5_decisions.py.")
