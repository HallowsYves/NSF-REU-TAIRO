"""
TAIRO-HX hierarchical level-chaining -- builds the actual Level 1->2->3->4
pipeline from the four previously-standalone classifiers, using genuine
out-of-fold (OOF) upstream predictions (evaluation/oof_chaining.py) so no
downstream level's training features leak in-sample confidence from an
upstream model's own fit. See CLAUDE.md's "Level-Chaining Architecture"
decision and this session's plan for the full design rationale.

Chain (immediate-predecessor conditioning only, confirmed 2026-07-20):

  Level 1 (task_stage)      <- base causal features only
  Level 2 source model      <- base causal features + Level 1 (onehot+conf)
    predicts failure_mode; p_fail = 1 - P(success); bucketed into
    normal/suspicious/abnormal/unknown via the SAME methodology as
    scripts/finalize_level2_labels.py (clean_2M/clean-condition p75/p95,
    recomputed fresh from this run's OOF p_fail -- not hardcoded, since the
    OOF procedure changes the distribution vs. the original in-sample scores
    in results/level2_pfail_scores.csv)
  Level 3 (failure_mode)    <- base causal features + Level 2 (onehot+conf)
  Level 4 (attack_family)   <- base causal features + Level 3 (onehot+conf)
                                (non-clean rows only, existing decision)

Level 2's "confidence" scalar is p_fail itself (not a predict_proba max) --
p_fail is already a single, informative, bounded [0,1] scalar directly
driving the bucketed label, more meaningful here than a proba max over an
intermediate failure_mode prediction that the bucketing itself discards.

Input:  results/classifier_seedfix/causal_feature_matrix.csv
        results/level1_labels.csv   (task_stage ground truth)
        results/level4_labels.csv   (attack_family ground truth)
Output: results/hierarchical_chain_predictions.csv
        KEY_COLS + failure_mode + attack_family + task_stage +
        level{1,3}_pred_class/confidence + level2_label/confidence(p_fail)
        + level4_pred_class/confidence (NaN for clean rows)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from evaluation.causal_features import FORBIDDEN_FEATURES
from evaluation.failure_mode_labeling import ALL_LABELS
from evaluation.oof_chaining import KEY_COLS, get_oof_predictions, onehot_upstream

OUT_PATH = "results/hierarchical_chain_predictions.csv"
TRAIN_SEEDS = [0, 1, 2, 3]
TEST_SEED = 4
EARLY_CUTOFF_T = 69  # mirrors scripts/finalize_level2_labels.py

# ── Load + merge ground truth ────────────────────────────────────────────
print("[chain] Loading causal feature matrix + Level 1/4 ground truth ...")
base_df = pd.read_csv("results/classifier_seedfix/causal_feature_matrix.csv")

NON_FEATURE_COLS = {"episode_idx", "checkpoint_t", "failure_mode", "condition",
                     "seed", "model", *FORBIDDEN_FEATURES}
BASE_FEATURE_COLS = [c for c in base_df.columns if c not in NON_FEATURE_COLS]
assert not any(f in BASE_FEATURE_COLS for f in FORBIDDEN_FEATURES)
print(f"[chain] Base causal features ({len(BASE_FEATURE_COLS)}): {BASE_FEATURE_COLS}\n")

level1_lab = pd.read_csv("results/level1_labels.csv")[KEY_COLS + ["task_stage"]]
level4_lab = pd.read_csv("results/level4_labels.csv")[KEY_COLS + ["attack_family"]]

df = base_df.merge(level1_lab, on=KEY_COLS, how="left")
assert df["task_stage"].isna().sum() == 0, "task_stage should never be NaN"
df = df.merge(level4_lab, on=KEY_COLS, how="left")
assert len(df) == len(base_df), "Merge changed row count -- KEY_COLS is not 1:1"
print(f"[chain] {len(df)} total rows\n")

# ── Level 1: task_stage <- base causal features ──────────────────────────
print("=" * 70)
print("LEVEL 1 -- task_stage (base causal features only)")
print("=" * 70)
l1_preds, l1_model, _ = get_oof_predictions(
    df, BASE_FEATURE_COLS, "task_stage", TRAIN_SEEDS, TEST_SEED, prefix="level1")
l1_onehot = onehot_upstream(l1_preds, "level1", config.LEVEL1_STAGES)
df = df.merge(l1_onehot, on=KEY_COLS, how="left")
L1_ONEHOT_COLS = [f"level1_{s}" for s in config.LEVEL1_STAGES] + ["level1_confidence"]

# ── Level 2 source model: failure_mode <- base features + Level 1 ───────
print("=" * 70)
print("LEVEL 2 source -- failure_mode <- base causal features + Level 1")
print("=" * 70)
L2_FEATURE_COLS = BASE_FEATURE_COLS + L1_ONEHOT_COLS
l2src_preds, l2src_model, l2src_proba = get_oof_predictions(
    df, L2_FEATURE_COLS, "failure_mode", TRAIN_SEEDS, TEST_SEED, prefix="level2src")

p_fail = 1.0 - l2src_proba["success"]
pfail_df = l2src_proba[KEY_COLS].copy()
pfail_df["p_fail"] = p_fail.values
df = df.merge(pfail_df, on=KEY_COLS, how="left")
assert df["p_fail"].isna().sum() == 0

clean_ref = df[(df["model"] == "clean_2M") & (df["condition"] == "clean")]["p_fail"].values
CLEAN_P75 = float(np.percentile(clean_ref, 75))
CLEAN_P95 = float(np.percentile(clean_ref, 95))
print(f"[chain] Level 2 thresholds (recomputed fresh on this run's OOF p_fail, "
      f"clean_2M/clean, n={len(clean_ref)}): P75={CLEAN_P75!r}  P95={CLEAN_P95!r}")


def _level2_label(row):
    if row["model"] != "clean_2M":
        return "unknown"
    if row["checkpoint_t"] < EARLY_CUTOFF_T:
        return "unknown"
    if row["p_fail"] < CLEAN_P75:
        return "normal"
    if row["p_fail"] < CLEAN_P95:
        return "suspicious"
    return "abnormal"


df["level2_pred_class"] = df.apply(_level2_label, axis=1)
LEVEL2_CLASSES = ["normal", "suspicious", "abnormal", "unknown"]
l2_preds_for_onehot = df[KEY_COLS + ["level2_pred_class"]].copy()
l2_preds_for_onehot["level2_confidence"] = df["p_fail"].values
l2_onehot = onehot_upstream(l2_preds_for_onehot, "level2", LEVEL2_CLASSES)
# l2_onehot already carries level2_confidence (= p_fail) -- do not also keep a
# separate df["level2_confidence"] pre-merge, or the merge would collide.
df = df.merge(l2_onehot, on=KEY_COLS, how="left")
L2_ONEHOT_COLS = [f"level2_{c}" for c in LEVEL2_CLASSES] + ["level2_confidence"]

print("\n[chain] Level 2 label distribution (this run, OOF-based):")
print(df["level2_pred_class"].value_counts().to_string())
print()

# ── Level 3: failure_mode <- base features + Level 2 ─────────────────────
print("=" * 70)
print("LEVEL 3 -- failure_mode <- base causal features + Level 2")
print("=" * 70)
L3_FEATURE_COLS = BASE_FEATURE_COLS + L2_ONEHOT_COLS
l3_preds, l3_model, _ = get_oof_predictions(
    df, L3_FEATURE_COLS, "failure_mode", TRAIN_SEEDS, TEST_SEED, prefix="level3")
l3_onehot = onehot_upstream(l3_preds, "level3", ALL_LABELS)
df = df.merge(l3_onehot, on=KEY_COLS, how="left")
L3_ONEHOT_COLS = [f"level3_{c}" for c in ALL_LABELS] + ["level3_confidence"]

# ── Level 4: attack_family <- base features + Level 3 (non-clean rows) ──
print("=" * 70)
print("LEVEL 4 -- attack_family <- base causal features + Level 3 (non-clean only)")
print("=" * 70)
L4_FEATURE_COLS = BASE_FEATURE_COLS + L3_ONEHOT_COLS
l4_preds, l4_model, _ = get_oof_predictions(
    df, L4_FEATURE_COLS, "attack_family", TRAIN_SEEDS, TEST_SEED, prefix="level4")

# ── Assemble final output ─────────────────────────────────────────────────
# level1_pred_class/level3_pred_class live only in l1_preds/l3_preds (df only
# kept their one-hot-encoded form for chaining) -- merge those back in here.
out = df[KEY_COLS + ["failure_mode", "attack_family", "task_stage",
                      "level2_pred_class", "level2_confidence"]].copy()
out = out.merge(l1_preds, on=KEY_COLS, how="left")
out = out.merge(l3_preds, on=KEY_COLS, how="left")
out = out.merge(l4_preds, on=KEY_COLS, how="left")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out.to_csv(OUT_PATH, index=False)
print(f"[chain] Saved -> {OUT_PATH}  ({len(out)} rows, {len(l4_preds)} with a Level 4 prediction)")
print("\n[chain] Done.")
