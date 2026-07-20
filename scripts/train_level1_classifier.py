"""
TAIRO-HX Level 1 (Task Stage) classifier -- trains + evaluates a Random
Forest that predicts `task_stage` from causal behavior features alone,
distinct from the deterministic ground-truth labeling in
scripts/build_level1_labels.py.

Standalone classifier only -- no upstream chaining (no Level 2/3/4 predicted
classes as inputs). Out-of-fold chaining plumbing doesn't exist yet, so full
hierarchical chaining stays separate, later work per CLAUDE.md's
Level-Chaining Architecture decision.

Same RF setup, seed split, and core eval suite as scripts/train_level4_classifier.py
-- reused closely where applicable. Two differences from that template
(both judgment calls, confirmed 2026-07-20):
  - ALL rows are trained on (6-class problem) -- unlike Level 4, task_stage
    has no "clean" gap: every row (attacked or not) is doing *something* in
    the task, so there is no analogous class to exclude.
  - Level 4's two attack-family-specific generalization checks don't port
    over as-is (task stage isn't tied to specific attack conditions the way
    attack family is). Replaced with a single condition-holdout check: hold
    out object_pose_spoof entirely and confirm task-stage detection still
    works on it, since stage should generalize across conditions.

Does NOT save a model yet: no downstream consumer, same reasoning Level 4's
build used.

Input:  results/level1_labels.csv (causal features + task_stage ground truth,
        already merged by scripts/build_level1_labels.py)
Output: results/classifier_level1/level1_eval_summary.csv
        results/classifier_level1/level1_per_class_report.csv
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

import config
from evaluation.causal_features import FORBIDDEN_FEATURES

_parser = argparse.ArgumentParser()
_parser.add_argument("--labels-path", type=str, default="results/level1_labels.csv")
_parser.add_argument("--out-dir", type=str, default="results/classifier_level1")
_args = _parser.parse_args()
OUT_DIR = _args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]

STAGES = config.LEVEL1_STAGES

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(_args.labels_path)
assert df["task_stage"].isna().sum() == 0, "task_stage should never be NaN -- all rows get a stage"
assert set(df["task_stage"].unique()) == set(STAGES), \
    f"Unexpected stage values: {set(df['task_stage'].unique()) - set(STAGES)}"
print(f"[level1-clf] Loaded {len(df)} rows, all trained (no class excluded -- see module docstring)")
print(f"[level1-clf] Stages: {STAGES}\n")

NON_FEATURE_COLS = {"episode_idx", "checkpoint_t", "failure_mode", "task_stage",
                    "condition", "seed", "model", *FORBIDDEN_FEATURES}
FEATURE_COLS = [c for c in df.columns if c not in NON_FEATURE_COLS]
assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES), \
    f"Forbidden feature leaked: {set(FEATURE_COLS) & FORBIDDEN_FEATURES}"
assert "task_stage" not in FEATURE_COLS and "failure_mode" not in FEATURE_COLS

print(f"[level1-clf] Features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")

train_df = df[df["seed"].isin(TRAIN_SEEDS)].copy()
test_df = df[df["seed"] == TEST_SEED].copy()

X_train, y_train = train_df[FEATURE_COLS].values, train_df["task_stage"].values
X_test, y_test = test_df[FEATURE_COLS].values, test_df["task_stage"].values

print(f"[level1-clf] Train: {len(X_train)} rows (seeds {TRAIN_SEEDS})")
print(f"[level1-clf] Test:  {len(X_test)} rows  (seed {TEST_SEED})\n")

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
train_counts = train_df["task_stage"].value_counts().reindex(STAGES, fill_value=0)
test_counts = test_df["task_stage"].value_counts().reindex(STAGES, fill_value=0)
counts_df = pd.DataFrame({"train": train_counts, "test": test_counts})
counts_df["train_pct"] = (100 * counts_df["train"] / counts_df["train"].sum()).round(2)
counts_df["test_pct"] = (100 * counts_df["test"] / counts_df["test"].sum()).round(2)
print(counts_df.to_string())
print()

# ── Majority-class baseline ──────────────────────────────────────────────
majority_label = train_df["task_stage"].value_counts().idxmax()
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
print("  (n_estimators=300, class_weight='balanced', seed=42 — same as Phase 8.5/9B/Level4)")
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
report_str = classification_report(y_test, y_pred, labels=STAGES, zero_division=0, digits=3)
print(report_str)
report_dict = classification_report(y_test, y_pred, labels=STAGES,
                                     zero_division=0, digits=3, output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv(os.path.join(OUT_DIR, "level1_per_class_report.csv"))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred, labels=STAGES)
cm_df = pd.DataFrame(cm, index=[f"T:{l[:16]}" for l in STAGES],
                              columns=[f"P:{l[:16]}" for l in STAGES])
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
      "episodes — rows from the same episode are correlated (14 checkpoints share a stage "
      "trajectory), so this CI is narrower than a true episode-level CI. Treat as indicative, "
      "not exact.")
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
    a = accuracy_score(sub["task_stage"], sub["pred"])
    f = f1_score(sub["task_stage"], sub["pred"], average="macro", zero_division=0)
    print(f"  t={t:>3}  n={len(sub):>5}  acc={a:.3f}  macro-F1={f:.3f}")
print()

# ── Generalization check: hold out object_pose_spoof entirely ────────────
print("=" * 70)
print("GENERALIZATION CHECK — held out object_pose_spoof")
print("  Unlike Level 4's attack-family classes, task_stage is not tied to a")
print("  specific attack condition -- every condition produces all 6 stages.")
print("  This checks whether stage detection still works when an entire")
print("  condition is unseen during training (real generalization test).")
print("=" * 70)

HOLDOUT = ["object_pose_spoof"]
gen_train = df[~df["condition"].isin(HOLDOUT)].copy()
gen_test = df[df["condition"].isin(HOLDOUT)].copy()

rf_gen = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf_gen.fit(gen_train[FEATURE_COLS].values, gen_train["task_stage"].values)
y_pred_gen = rf_gen.predict(gen_test[FEATURE_COLS].values)
y_true_gen = gen_test["task_stage"].values

acc_gen = accuracy_score(y_true_gen, y_pred_gen)
f1_gen = f1_score(y_true_gen, y_pred_gen, average="macro", zero_division=0)
print(f"  Train: {len(gen_train)} rows  Test: {len(gen_test)} rows  (held out: {HOLDOUT})")
print(f"  Accuracy: {acc_gen:.3f}")
print(f"  Macro F1: {f1_gen:.3f}\n")
present_gen = [l for l in STAGES if l in y_true_gen]
print(classification_report(y_true_gen, y_pred_gen, labels=present_gen, zero_division=0, digits=3))
print()

# ── Save summary CSV ──────────────────────────────────────────────────────
summary = pd.DataFrame([{
    "n_rows_total": len(df), "n_train": len(X_train), "n_test": len(X_test),
    "baseline_accuracy": baseline_acc, "baseline_macro_f1": baseline_f1,
    "accuracy": acc, "macro_f1": macro_f1, "balanced_accuracy": bal_acc,
    "macro_f1_ci_lo": ci_lo, "macro_f1_ci_hi": ci_hi,
    "gen_check_holdout": ",".join(HOLDOUT), "gen_check_accuracy": acc_gen, "gen_check_macro_f1": f1_gen,
}])
summary.to_csv(os.path.join(OUT_DIR, "level1_eval_summary.csv"), index=False)
print(f"[level1-clf] Saved summary -> {os.path.join(OUT_DIR, 'level1_eval_summary.csv')}")
print(f"[level1-clf] Saved per-class report -> {os.path.join(OUT_DIR, 'level1_per_class_report.csv')}")
print("\n[level1-clf] Done. No model saved (no downstream consumer yet).")
