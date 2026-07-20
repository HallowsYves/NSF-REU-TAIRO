"""
TAIRO-HX Level 4 (Attack Family) ground-truth labeling.

Decision locked in 2026-07-20 (see CLAUDE.md "Level 4 (Attack Family)
Labeling — standing decision"):

  - Level 4's 5-class taxonomy (TAIRO-HX.md Section 3) is a deterministic
    lookup from the already-known `condition` column -- no model fitting.
    This is a ground-truth LABEL for training/evaluating a future Level 4
    classifier, not the classifier itself (which would predict attack
    family from behavior features, without access to `condition`).
  - `attack_family` is populated for ALL non-clean rows regardless of
    Level 2's predicted verdict (`level2_label`). Level 2 gating (which
    rows are actually usable, given Level 2's clean_2M-only scope) is
    deferred to whenever the Level 4 classifier itself is trained, not
    baked into this label file.
  - sensor_bias maps to "sensor_info_loss" (grouped with sensor_dropout/
    contact_dropout), not "perception_state" -- conservative reading, only
    object_pose_spoof moves out of the old 4-class map's "sensor" bucket.

Input:  results/level2_labels_clean_2M_scoped.csv
Output: results/level4_labels.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config

IN_PATH = "results/level2_labels_clean_2M_scoped.csv"
OUT_PATH = "results/level4_labels.csv"

df = pd.read_csv(IN_PATH)

df["attack_family"] = df["condition"].map(config.ATTACK_FAMILY_MAP)

df.to_csv(OUT_PATH, index=False)
print(f"[level4-labels] Saved -> {OUT_PATH}  ({len(df)} rows)\n")

print("=" * 70)
print("ATTACK_FAMILY DISTRIBUTION (all rows, all models/conditions/checkpoints)")
print("=" * 70)
counts = df["attack_family"].value_counts(dropna=False)
pct = (100 * counts / len(df)).round(2)
dist_df = pd.DataFrame({"count": counts, "pct": pct})
print(dist_df.to_string())
print()

print("=" * 70)
print("CROSS-TAB: attack_family x level2_label (sanity check only, not a filter)")
print("=" * 70)
crosstab = pd.crosstab(df["attack_family"], df["level2_label"], dropna=False)
print(crosstab.to_string())

print("\n[level4-labels] Done.")
