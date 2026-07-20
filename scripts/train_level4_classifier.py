"""
TAIRO-HX Level 4 (Attack Family) classifier — trains + evaluates a Random
Forest that predicts `attack_family` from causal behavior features alone
(no `condition`), distinct from the deterministic ground-truth labeling in
scripts/build_level4_labels.py.

Standalone classifier only -- no upstream chaining (no Level 2/3 predicted
classes as inputs). Level 3's own classifier and the out-of-fold chaining
plumbing don't exist yet, so full hierarchical chaining stays separate,
later work per CLAUDE.md's Level-Chaining Architecture decision.

Same RF setup, seed split, and eval suite as scripts/train_causal_classifier.py
(Phase 9B) -- reused verbatim where applicable. Does NOT save a model yet:
this build has no downstream consumer, same reasoning Phase 9B used before
Phase 9C's production save.

Clean rows (attack_family is NaN) are EXCLUDED, not trained as a 5th class
(decided 2026-07-20, see CLAUDE.md "Level 4 (Attack Family) Labeling"):
Level 4 has no "clean" class in the memo, and clean-vs-not is Level 2's job.
`unknown_attack` is also never trained on -- it has zero ground-truth rows
(never assigned to any of the 11 conditions in ATTACK_FAMILY_MAP), so it
remains a theoretical prediction-time fallback only.

Input:  results/classifier_seedfix/causal_feature_matrix.csv (features)
        results/level4_labels.csv (attack_family ground truth)
Output: results/classifier_level4/level4_eval_summary.csv
        results/classifier_level4/level4_per_class_report.csv
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              f1_score, balanced_accuracy_score)

from config import CLASSIFIER_DIR as _DEFAULT_CLASSIFIER_DIR
from evaluation.causal_features import FORBIDDEN_FEATURES

_parser = argparse.ArgumentParser()
_parser.add_argument("--classifier-dir", type=str, default=None,
                      help="Dir containing causal_feature_matrix.csv (default: config.CLASSIFIER_DIR)")
_parser.add_argument("--out-dir", type=str, default="results/classifier_level4")
_args = _parser.parse_args()
CLASSIFIER_DIR = _args.classifier_dir if _args.classifier_dir is not None else _DEFAULT_CLASSIFIER_DIR
OUT_DIR = _args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]

TRAINED_FAMILIES = ["action_actuation", "perception_state", "goal_manipulation", "sensor_info_loss"]

KEY_COLS = ["model", "condition", "seed", "episode_idx", "checkpoint_t"]

# ── Load + merge ground-truth labels onto causal features ───────────────────
feat_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "causal_feature_matrix.csv"))
lab_df = pd.read_csv("results/level4_labels.csv")[KEY_COLS + ["attack_family"]]
merged = feat_df.merge(lab_df, on=KEY_COLS, how="left")
assert len(merged) == len(feat_df), "Merge changed row count -- composite key is not 1:1"
n_clean = merged["attack_family"].isna().sum()
assert n_clean == 8400, f"Expected exactly 8400 clean/unlabeled rows, got {n_clean}"

df = merged[merged["attack_family"].notna()].copy()
print(f"[level4-clf] Loaded {len(merged)} rows, excluded {n_clean} clean rows -> {len(df)} rows")
print(f"[level4-clf] Trained families: {TRAINED_FAMILIES}\n")

NON_FEATURE_COLS = {"episode_idx", "checkpoint_t", "failure_mode", "attack_family",
                    "condition", "seed", "model", *FORBIDDEN_FEATURES}
FEATURE_COLS = [c for c in df.columns if c not in NON_FEATURE_COLS]
assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES), \
    f"Forbidden feature leaked: {set(FEATURE_COLS) & FORBIDDEN_FEATURES}"
assert "attack_family" not in FEATURE_COLS and "failure_mode" not in FEATURE_COLS

print(f"[level4-clf] Features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")

train_df = df[df["seed"].isin(TRAIN_SEEDS)].copy()
test_df = df[df["seed"] == TEST_SEED].copy()

X_train, y_train = train_df[FEATURE_COLS].values, train_df["attack_family"].values
X_test, y_test = test_df[FEATURE_COLS].values, test_df["attack_family"].values

print(f"[level4-clf] Train: {len(X_train)} rows (seeds {TRAIN_SEEDS})")
print(f"[level4-clf] Test:  {len(X_test)} rows  (seed {TEST_SEED})\n")

# ── Leakage check ─────────────────────────────────────────────────────────
print("=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)
train_ep = set(train_df["episode_idx"])
test_ep = set(test_df["episode_idx"])
ep_overlap = train_ep & test_ep
print(f"  episode_idx overlap between train/test: {len(ep_overlap)} (expect 0)")
assert len(ep_overlap) == 0, f"LEAKAGE: {len(ep_overlap)} episode_idx values appear in both train and test"
train_pairs = set(zip(train_df["episode_idx"], train_df["checkpoint_t"]))
test_pairs = set(zip(test_df["episode_idx"], test_df["checkpoint_t"]))
print(f"  (episode_idx, checkpoint_t) row overlap: {len(train_pairs & test_pairs)} (expect 0)")
print()

# ── Class counts per split ────────────────────────────────────────────────
print("=" * 70)
print("CLASS COUNTS PER SPLIT (row-level, i.e. per checkpoint, not per episode)")
print("=" * 70)
train_counts = train_df["attack_family"].value_counts().reindex(TRAINED_FAMILIES, fill_value=0)
test_counts = test_df["attack_family"].value_counts().reindex(TRAINED_FAMILIES, fill_value=0)
counts_df = pd.DataFrame({"train": train_counts, "test": test_counts})
counts_df["train_pct"] = (100 * counts_df["train"] / counts_df["train"].sum()).round(2)
counts_df["test_pct"] = (100 * counts_df["test"] / counts_df["test"].sum()).round(2)
print(counts_df.to_string())
print()

# ── Majority-class baseline ──────────────────────────────────────────────
majority_label = train_df["attack_family"].value_counts().idxmax()
y_majority = np.full(len(y_test), majority_label)
baseline_acc = accuracy_score(y_test, y_majority)
baseline_f1 = f1_score(y_test, y_majority, average="macro", zero_division=0)
print("=" * 70)
print(f"BASELINE — majority class (always predicts '{majority_label}')")
print("=" * 70)
print(f"  Accuracy: {baseline_acc:.3f}")
print(f"  Macro F1: {baseline_f1:.3f}\n")

# ── Random Forest (pooled — all checkpoints together) ────────────────────
print("=" * 70)
print("RANDOM FOREST — CAUSAL FEATURES, POOLED ACROSS ALL CHECKPOINTS")
print("  (n_estimators=300, class_weight='balanced', seed=42 — same as Phase 8.5/9B)")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
print(f"  Accuracy: {acc:.3f}   (majority baseline: {baseline_acc:.3f}, delta={acc-baseline_acc:+.3f})")
print(f"  Macro F1: {macro_f1:.3f}   (majority baseline: {baseline_f1:.3f}, delta={macro_f1-baseline_f1:+.3f})\n")

print("Per-class report:")
report_str = classification_report(y_test, y_pred, labels=TRAINED_FAMILIES, zero_division=0, digits=3)
print(report_str)
report_dict = classification_report(y_test, y_pred, labels=TRAINED_FAMILIES,
                                     zero_division=0, digits=3, output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv(os.path.join(OUT_DIR, "level4_per_class_report.csv"))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred, labels=TRAINED_FAMILIES)
cm_df = pd.DataFrame(cm, index=[f"T:{l[:16]}" for l in TRAINED_FAMILIES],
                              columns=[f"P:{l[:16]}" for l in TRAINED_FAMILIES])
print(cm_df.to_string())
print()

# ── Balanced accuracy + bootstrap CI for macro-F1 ────────────────────────
print("=" * 70)
print("BALANCED ACCURACY + MACRO-F1 95% CI (bootstrap, n=2000 resamples)")
print("=" * 70)
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"  Balanced accuracy: {bal_acc:.3f}")
rng = np.random.default_rng(42)
n_test = len(y_test)
y_test_arr = np.asarray(y_test)
y_pred_arr = np.asarray(y_pred)
boot_f1 = []
for _ in range(2000):
    idx = rng.integers(0, n_test, n_test)
    boot_f1.append(f1_score(y_test_arr[idx], y_pred_arr[idx], average="macro", zero_division=0))
boot_f1 = np.array(boot_f1)
ci_lo, ci_hi = np.percentile(boot_f1, [2.5, 97.5])
print(f"  Macro-F1: {macro_f1:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
print("  NOTE: bootstrap resamples individual (episode_idx, checkpoint_t) rows, not whole "
      "episodes — rows from the same episode are correlated (14 checkpoints share one label), "
      "so this CI is narrower than a true episode-level CI. Treat as indicative, not exact.")
print()

# ── Feature importances ──────────────────────────────────────────────────
print("=" * 70)
print("FEATURE IMPORTANCES (top 20)")
print("=" * 70)
imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat, val in imp.head(20).items():
    bar = "█" * int(val * 300)
    print(f"  {feat:<30}  {val:.4f}  {bar}")
print()

# ── Accuracy / macro-F1 as a function of checkpoint_t ────────────────────
print("=" * 70)
print("ACCURACY / MACRO-F1 BY CHECKPOINT")
print("=" * 70)
test_df = test_df.copy()
test_df["pred"] = y_pred
for t, sub in test_df.groupby("checkpoint_t"):
    a = accuracy_score(sub["attack_family"], sub["pred"])
    f = f1_score(sub["attack_family"], sub["pred"], average="macro", zero_division=0)
    print(f"  t={t:>3}  n={len(sub):>5}  acc={a:.3f}  macro-F1={f:.3f}")
print()

# ── Generalization check 1: genuine cross-condition generalization ───────
print("=" * 70)
print("GENERALIZATION CHECK 1 — held out action_delay + sensor_dropout")
print("  (both families retain other backing conditions -- real generalization test)")
print("=" * 70)

HOLDOUT_1 = ["action_delay", "sensor_dropout"]
gen1_train = df[~df["condition"].isin(HOLDOUT_1)].copy()
gen1_test = df[df["condition"].isin(HOLDOUT_1)].copy()

rf_gen1 = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf_gen1.fit(gen1_train[FEATURE_COLS].values, gen1_train["attack_family"].values)
y_pred_gen1 = rf_gen1.predict(gen1_test[FEATURE_COLS].values)
y_true_gen1 = gen1_test["attack_family"].values

acc_gen1 = accuracy_score(y_true_gen1, y_pred_gen1)
f1_gen1 = f1_score(y_true_gen1, y_pred_gen1, average="macro", zero_division=0)
print(f"  Train: {len(gen1_train)} rows  Test: {len(gen1_test)} rows  (held out: {HOLDOUT_1})")
print(f"  Accuracy: {acc_gen1:.3f}")
print(f"  Macro F1: {f1_gen1:.3f}\n")
present_1 = [l for l in TRAINED_FAMILIES if l in y_true_gen1]
print(classification_report(y_true_gen1, y_pred_gen1, labels=present_1, zero_division=0, digits=3))
print()

# ── Generalization check 2: single-condition fragility demo ──────────────
print("=" * 70)
print("GENERALIZATION CHECK 2 — held out object_pose_spoof (fragility demo)")
print("  perception_state has only ONE backing condition (object_pose_spoof).")
print("  Holding it out leaves ZERO training examples for that class -- expect")
print("  perception_state F1 = 0.000. This is a structural property of a")
print("  single-condition class, not a surprising classifier failure.")
print("=" * 70)

HOLDOUT_2 = ["object_pose_spoof"]
gen2_train = df[~df["condition"].isin(HOLDOUT_2)].copy()
gen2_test = df[df["condition"].isin(HOLDOUT_2)].copy()

rf_gen2 = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf_gen2.fit(gen2_train[FEATURE_COLS].values, gen2_train["attack_family"].values)
y_pred_gen2 = rf_gen2.predict(gen2_test[FEATURE_COLS].values)
y_true_gen2 = gen2_test["attack_family"].values

acc_gen2 = accuracy_score(y_true_gen2, y_pred_gen2)
f1_perception_gen2 = f1_score(y_true_gen2, y_pred_gen2, labels=["perception_state"],
                               average="macro", zero_division=0)
print(f"  Train: {len(gen2_train)} rows  Test: {len(gen2_test)} rows  (held out: {HOLDOUT_2})")
print(f"  Accuracy on held-out perception_state rows: {acc_gen2:.3f}")
print(f"  perception_state F1: {f1_perception_gen2:.3f}  (expected 0.000)\n")

# ── Save summary CSV ──────────────────────────────────────────────────────
summary = pd.DataFrame([{
    "n_rows_total": len(merged), "n_rows_clean_excluded": n_clean, "n_rows_trained": len(df),
    "n_train": len(X_train), "n_test": len(X_test),
    "baseline_accuracy": baseline_acc, "baseline_macro_f1": baseline_f1,
    "accuracy": acc, "macro_f1": macro_f1, "balanced_accuracy": bal_acc,
    "macro_f1_ci_lo": ci_lo, "macro_f1_ci_hi": ci_hi,
    "gen_check1_holdout": ",".join(HOLDOUT_1), "gen_check1_accuracy": acc_gen1, "gen_check1_macro_f1": f1_gen1,
    "gen_check2_holdout": ",".join(HOLDOUT_2), "gen_check2_perception_state_f1": f1_perception_gen2,
}])
summary.to_csv(os.path.join(OUT_DIR, "level4_eval_summary.csv"), index=False)
print(f"[level4-clf] Saved summary -> {os.path.join(OUT_DIR, 'level4_eval_summary.csv')}")
print(f"[level4-clf] Saved per-class report -> {os.path.join(OUT_DIR, 'level4_per_class_report.csv')}")
print("\n[level4-clf] Done. No model saved (no downstream consumer yet).")
