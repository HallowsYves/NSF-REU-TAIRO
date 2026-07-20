"""
TAIRO-HX Level 1 (Task Stage) ground-truth labeling.

Decision locked in 2026-07-20 (see CLAUDE.md "Level 1 (Task Stage) Labeling
— standing decision"):

  - Level 1's 6-stage taxonomy (TAIRO-HX.md Section 3) is a deterministic,
    priority-ordered rule cascade over existing causal (non-hindsight)
    features -- no model fitting. Mirrors label_episode()'s elif structure
    in evaluation/failure_mode_labeling.py, but applied per-checkpoint-row
    rather than per-episode.
  - Approach->Align reuses REACH_THRESHOLD via `reached_ever_sofar` -- no
    new constant. Transport->Place uses the new PROVISIONAL
    config.LEVEL1_PLACE_RADIUS (0.10m); flagged as uncalibrated.
  - No 7th "stalled/failed" class: episodes that never reach/grasp/arrive
    simply stay at their last-attempted stage across all 14 checkpoints.
    grasped_but_dropped episodes stay "transporting" after the drop
    (grasp_kinematic_ever_sofar is sticky) rather than regressing to
    "grasping" -- the drop itself is Level 3's ( failure_mode's) signal.

Input:  results/classifier_seedfix/causal_feature_matrix.csv
Output: results/level1_labels.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config

IN_PATH = "results/classifier_seedfix/causal_feature_matrix.csv"
OUT_PATH = "results/level1_labels.csv"

KEY_COLS = ["model", "condition", "seed", "episode_idx", "checkpoint_t"]

STAGE_ORDINAL = {stage: i + 1 for i, stage in enumerate(config.LEVEL1_STAGES)}

df = pd.read_csv(IN_PATH)

conditions = [
    df["is_success_now"] == 1,
    (df["grasp_kinematic_ever_sofar"] == 1) & (df["dttg_now"] < config.LEVEL1_PLACE_RADIUS),
    df["grasp_kinematic_ever_sofar"] == 1,
    df["contact_streak_now"] >= 1,
    df["reached_ever_sofar"] == 1,
]
choices = [
    "verifying_completion",
    "placing",
    "transporting",
    "grasping",
    "aligning_gripper",
]
df["task_stage"] = np.select(conditions, choices, default="approaching_object")

df.to_csv(OUT_PATH, index=False)
print(f"[level1-labels] Saved -> {OUT_PATH}  ({len(df)} rows)\n")

print("=" * 70)
print("TASK_STAGE DISTRIBUTION (all rows, all models/conditions/checkpoints)")
print("=" * 70)
counts = df["task_stage"].value_counts()
pct = (100 * counts / len(df)).round(2)
dist_df = pd.DataFrame({"count": counts, "pct": pct}).reindex(config.LEVEL1_STAGES)
print(dist_df.to_string())
print()

print("=" * 70)
print("CROSS-TAB: task_stage x checkpoint_t (mean checkpoint_t per stage)")
print("=" * 70)
mean_t = df.groupby("task_stage")["checkpoint_t"].agg(["mean", "count"]).reindex(config.LEVEL1_STAGES)
print(mean_t.to_string())
print()

print("=" * 70)
print("PER-EPISODE MONOTONICITY CHECK")
print("=" * 70)
df["_ordinal"] = df["task_stage"].map(STAGE_ORDINAL)
df_sorted = df.sort_values(KEY_COLS[:-1] + ["checkpoint_t"])
df_sorted["_diff"] = df_sorted.groupby(KEY_COLS[:-1])["_ordinal"].diff()
regressed_rows = (df_sorted["_diff"] < 0).sum()
n_episodes = df_sorted.groupby(KEY_COLS[:-1]).ngroups
regressed_episodes = df_sorted[df_sorted["_diff"] < 0].groupby(KEY_COLS[:-1]).ngroups
print(f"Rows with a stage-ordinal decrease vs. previous checkpoint: {regressed_rows} / {len(df_sorted)}")
print(f"Episodes with at least one regression: {regressed_episodes} / {n_episodes} "
      f"({100 * regressed_episodes / n_episodes:.2f}%)")
print()

print("=" * 70)
print("CROSS-TAB: task_stage x failure_mode (invariant checks)")
print("=" * 70)
crosstab = pd.crosstab(df["task_stage"], df["failure_mode"]).reindex(config.LEVEL1_STAGES)
print(crosstab.to_string())
print()

adv_stages = ["grasping", "transporting", "placing", "verifying_completion"]
never_reached_violations = df[
    (df["failure_mode"] == "never_reached_object") & (df["task_stage"].isin(adv_stages))
]
print(f"[invariant] never_reached_object rows reaching grasping+: {len(never_reached_violations)} "
      f"(expect 0)")

transport_plus = ["transporting", "placing", "verifying_completion"]
failed_grasp_violations = df[
    (df["failure_mode"] == "reached_but_failed_grasp") & (df["task_stage"].isin(transport_plus))
]
print(f"[invariant] reached_but_failed_grasp rows reaching transporting+: {len(failed_grasp_violations)} "
      f"(expect 0)")

spoofed_goal_violations = df[
    (df["failure_mode"] == "spoofed_goal") & (df["task_stage"] == "verifying_completion")
]
print(f"[invariant] spoofed_goal rows reaching verifying_completion: {len(spoofed_goal_violations)} "
      f"(expect 0)")

print("\n[level1-labels] Done.")
