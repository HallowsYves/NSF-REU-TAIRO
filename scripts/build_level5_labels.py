"""
TAIRO-HX Level 5 (Recovery Decision) labeling.

Decision locked in 2026-07-20 (see CLAUDE.md "Level-Chaining Architecture"
and this session's design sign-off):

  - Level 5 is RULE-BASED decision logic over Levels 2-4's CHAINED
    PREDICTIONS (level2_pred_class, level3_pred_class == failure_mode,
    level4_pred_class == attack_family, from
    results/hierarchical_chain_predictions.csv), not ground-truth labels
    and not a learned classifier. Level 1 is deliberately excluded (per
    CLAUDE.md, Level 5 consumes Levels 2-4 only).
  - Output taxonomy is the memo's 7-decision scheme (TAIRO-HX.md Section 3,
    "Level 5: Recoverability"), not Section 6's 5 recovery families —
    Section 3 is Level 5's own defined output; Section 6 is a
    controller-side reorganization of recovery_v4.py's specialist rules,
    which this session keeps deliberately separate (sign-off: Level 5 is a
    new analysis artifact, not wired into recovery_v4.py or
    episode_runner.py this session).
  - Built across all 4 model checkpoints (mirrors Level 2's own labeling
    pattern): outside clean_2M, level2_pred_class skews toward "unknown"
    (Level 2's documented clean_2M-only scope), so Level 5's decisions
    there will skew toward the continue_sac_her default — expected
    degeneracy, not a bug, and not hidden by restricting the output.
  - Two of the memo's three safe-stop triggers ("unknown attacks with high
    uncertainty", "repeated failed recovery attempts") are approximated
    with PROVISIONAL, uncalibrated proxies (LEVEL5_LOW_CONFIDENCE_THRESH,
    LEVEL5_ABNORMAL_STREAK_THRESH in config.py). The third ("loss of both
    vision and contact") has no implementable analog in this simulation —
    there is no camera/perception channel independent of oracle state (see
    RECOVERY_V4.md 2.5) — and is folded into the low-confidence proxy
    rather than invented.

Priority cascade (first match wins), evaluated per checkpoint row:
  1. level3_pred_class == "success"                     -> continue_sac_her
  2. level2_pred_class == "abnormal" AND both level3/4
     confidence < LEVEL5_LOW_CONFIDENCE_THRESH           -> stop_safely
  3. abnormal_streak_sofar >= LEVEL5_ABNORMAL_STREAK_THRESH -> stop_safely
  4. level3_pred_class == "spoofed_goal"                 -> restore_trusted_goal
  5. level3_pred_class in {never_reached_object,
     reached_but_failed_grasp, grasped_but_dropped}       -> retry_task_stage
  6. level3_pred_class == "divergent_transport" AND
     level4_pred_class == "action_actuation"              -> compensate_for_problem
  7. level4_pred_class in {perception_state,
     sensor_info_loss}                                    -> reconstruct_state
  8. level2_pred_class == "suspicious"                    -> continue_reduced_speed
  9. default (level2_pred_class in {normal, unknown})     -> continue_sac_her

`abnormal_streak_sofar` is causal: for each row it counts consecutive PRIOR
checkpoints (strictly earlier checkpoint_t, same model/seed/episode_idx)
with level2_pred_class == "abnormal", ending immediately before the current
row. Does not include the current row's own level2_pred_class.

Input:  results/hierarchical_chain_predictions.csv
Output: results/level5_decisions.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config

IN_PATH = "results/hierarchical_chain_predictions.csv"
OUT_PATH = "results/level5_decisions.csv"

TASK_RETRY_CLASSES = {
    "never_reached_object",
    "reached_but_failed_grasp",
    "grasped_but_dropped",
}
RECONSTRUCT_FAMILIES = {"perception_state", "sensor_info_loss"}

df = pd.read_csv(IN_PATH)
df = df.sort_values(["model", "seed", "episode_idx", "checkpoint_t"]).reset_index(drop=True)

# Causal, per-episode consecutive-abnormal-checkpoint streak (strictly prior
# checkpoints only -- never includes the current row).
group_cols = ["model", "seed", "episode_idx"]
is_abnormal = (df["level2_pred_class"] == "abnormal").astype(int)


def _prior_streak(s: pd.Series) -> pd.Series:
    # Running consecutive-True streak, then shifted by one row so the
    # current row's own value is excluded (causal: only strictly-prior
    # checkpoints count).
    streak = s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    streak = streak.where(s == 1, 0)
    return streak.shift(1, fill_value=0)


df["abnormal_streak_sofar"] = (
    is_abnormal.groupby([df[c] for c in group_cols]).transform(_prior_streak)
)


def decide(row) -> str:
    if row["level3_pred_class"] == "success":
        return "continue_sac_her"
    if (
        row["level2_pred_class"] == "abnormal"
        and row["level3_confidence"] < config.LEVEL5_LOW_CONFIDENCE_THRESH
        and row["level4_confidence"] < config.LEVEL5_LOW_CONFIDENCE_THRESH
    ):
        return "stop_safely"
    if row["abnormal_streak_sofar"] >= config.LEVEL5_ABNORMAL_STREAK_THRESH:
        return "stop_safely"
    if row["level3_pred_class"] == "spoofed_goal":
        return "restore_trusted_goal"
    if row["level3_pred_class"] in TASK_RETRY_CLASSES:
        return "retry_task_stage"
    if (
        row["level3_pred_class"] == "divergent_transport"
        and row["level4_pred_class"] == "action_actuation"
    ):
        return "compensate_for_problem"
    if row["level4_pred_class"] in RECONSTRUCT_FAMILIES:
        return "reconstruct_state"
    if row["level2_pred_class"] == "suspicious":
        return "continue_reduced_speed"
    return "continue_sac_her"


# level4_pred_class is NaN for clean/success rows (Level 4 excludes clean
# rows, same as results/level4_labels.csv) -- fillna so the `in` checks
# above behave, without altering the semantic meaning (NaN never matches
# either set).
df["level4_pred_class"] = df["level4_pred_class"].fillna("")

df["level5_decision"] = df.apply(decide, axis=1)

out_cols = [
    "model", "condition", "seed", "episode_idx", "checkpoint_t",
    "level2_pred_class", "level2_confidence",
    "level3_pred_class", "level3_confidence",
    "level4_pred_class", "level4_confidence",
    "abnormal_streak_sofar", "level5_decision",
]
df[out_cols].to_csv(OUT_PATH, index=False)
print(f"[level5-labels] Saved -> {OUT_PATH}  ({len(df)} rows)\n")

print("=" * 70)
print("LEVEL5_DECISION DISTRIBUTION (all rows, all models/conditions/checkpoints)")
print("=" * 70)
counts = df["level5_decision"].value_counts(dropna=False)
pct = (100 * counts / len(df)).round(2)
print(pd.DataFrame({"count": counts, "pct": pct}).to_string())
print()

print("=" * 70)
print("LEVEL5_DECISION DISTRIBUTION, clean_2M only")
print("=" * 70)
sub = df[df["model"] == "clean_2M"]
counts = sub["level5_decision"].value_counts(dropna=False)
pct = (100 * counts / len(sub)).round(2)
print(pd.DataFrame({"count": counts, "pct": pct}).to_string())
print()

print("=" * 70)
print("CROSS-TAB: level5_decision x model (sanity check -- expect degeneracy")
print("outside clean_2M, per Level 2's documented scope limitation)")
print("=" * 70)
print(pd.crosstab(df["level5_decision"], df["model"]).to_string())
print()

print("=" * 70)
print("CROSS-TAB: level5_decision x condition (clean_2M only)")
print("=" * 70)
print(pd.crosstab(sub["level5_decision"], sub["condition"]).to_string())

print("\n[level5-labels] Done.")
