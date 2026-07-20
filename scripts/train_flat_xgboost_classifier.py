"""
Item 1 Phase 1a (B1) — Genuine episode-level flat XGBoost baseline.

Same 34-column episode-level feature matrix, seed 0-3/4 train/test split, and
FORBIDDEN_FEATURES assertion as Phase 8 (train_failure_classifier.py). Only
the estimator changes: XGBClassifier in place of RandomForestClassifier.

Note on input: this reads the already-built
`results/classifier_seedfix/feature_matrix.csv` directly rather than
re-running build_features() over raw step logs — the per-episode step logs
for clean_500k / randomized_500k / randomized_2M are not present in this
repo checkout (only clean_2M step logs exist, under
results/data_recovery_v4*/), so the Phase 8 pipeline cannot be re-run from
scratch for all 4 models. feature_matrix.csv is the exact, unmodified output
of that pipeline (same 6600 rows used to produce the canonical 0.9424/0.8159
RF numbers), so reusing it preserves feature parity and split discipline
without inventing new numbers from a different feature set.

Output
------
  results/classifier_flat_xgboost/flat_xgboost_classifier.pkl
  results/classifier_flat_xgboost/feature_matrix.csv   (copy, for self-containment)
  Console: accuracy, macro-F1, per-class report, confusion matrix vs. Phase 8 RF baseline
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, balanced_accuracy_score)

from evaluation.failure_mode_labeling import ALL_LABELS

_parser = argparse.ArgumentParser()
_parser.add_argument("--feature-matrix", type=str,
                      default="results/classifier_seedfix/feature_matrix.csv",
                      help="Path to the existing Phase 8 episode-level feature matrix")
_parser.add_argument("--out-dir", type=str, default="results/classifier_flat_xgboost")
_args = _parser.parse_args()

os.makedirs(_args.out_dir, exist_ok=True)

TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]
FORBIDDEN_FEATURES = {"condition", "attack_level", "method"}

# ── Load existing episode-level feature matrix ────────────────────────────
print("[flat-xgb] Loading existing Phase 8 feature matrix ...")
feat_df = pd.read_csv(_args.feature_matrix)
print(f"  {len(feat_df)} episodes, {feat_df.shape[1]} cols\n")

feat_df.to_csv(os.path.join(_args.out_dir, "feature_matrix.csv"), index=False)

NON_FEATURE_COLS = {"episode_idx", "failure_mode", "condition", "seed", "model",
                    *FORBIDDEN_FEATURES}
FEATURE_COLS = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]

assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES), \
    f"Forbidden feature leaked: {set(FEATURE_COLS) & FORBIDDEN_FEATURES}"
print(f"[flat-xgb] Features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")

# ── Train / test split by seed (identical to Phase 8) ─────────────────────
train_df = feat_df[feat_df["seed"].isin(TRAIN_SEEDS)].copy()
test_df  = feat_df[feat_df["seed"] == TEST_SEED].copy()

X_train = train_df[FEATURE_COLS].values
X_test  = test_df[FEATURE_COLS].values

le = LabelEncoder()
le.fit(ALL_LABELS)
y_train = le.transform(train_df["failure_mode"].values)
y_test  = le.transform(test_df["failure_mode"].values)

print(f"[flat-xgb] Train: {len(X_train)} episodes (seeds {TRAIN_SEEDS})")
print(f"[flat-xgb] Test:  {len(X_test)} episodes  (seed {TEST_SEED})\n")

train_idx = set(train_df["episode_idx"])
test_idx  = set(test_df["episode_idx"])
overlap = train_idx & test_idx
assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} episode_idx values appear in both train and test"
print(f"[flat-xgb] Leakage check: episode_idx overlap = {len(overlap)} (expect 0)\n")

# ── XGBoost — reasonable defaults, not tuned (baseline comparison point) ──
print("=" * 70)
print("XGBOOST  (n_estimators=300, max_depth=6, learning_rate=0.1, seed=42)")
print("=" * 70)

# Class weighting: XGBClassifier has no built-in class_weight="balanced"
# equivalent for multiclass, so pass per-sample weights inversely
# proportional to class frequency (mirrors RF's class_weight="balanced").
class_counts = pd.Series(y_train).value_counts()
n_classes = len(class_counts)
n_samples = len(y_train)
class_weight_map = {c: n_samples / (n_classes * cnt) for c, cnt in class_counts.items()}
sample_weight = np.array([class_weight_map[c] for c in y_train])

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=len(ALL_LABELS),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
xgb.fit(X_train, y_train, sample_weight=sample_weight)
y_pred_xgb = xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb  = f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)
print(f"  Accuracy: {acc_xgb:.4f}")
print(f"  Macro F1: {f1_xgb:.4f}")
print()
print("Per-class report:")
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred_xgb),
                             labels=ALL_LABELS, zero_division=0, digits=3))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred_xgb, labels=list(range(len(ALL_LABELS))))
cm_df = pd.DataFrame(cm, index=[f"T:{l[:12]}" for l in le.inverse_transform(list(range(len(ALL_LABELS))))],
                              columns=[f"P:{l[:12]}" for l in le.inverse_transform(list(range(len(ALL_LABELS))))])
print(cm_df.to_string())
print()

bal_acc = balanced_accuracy_score(y_test, y_pred_xgb)
rng = np.random.default_rng(42)
n_test = len(y_test)
boot_f1 = []
y_test_arr = np.asarray(y_test)
y_pred_arr = np.asarray(y_pred_xgb)
for _ in range(2000):
    idx = rng.integers(0, n_test, n_test)
    boot_f1.append(f1_score(y_test_arr[idx], y_pred_arr[idx], average="macro", zero_division=0))
boot_f1 = np.array(boot_f1)
ci_lo, ci_hi = np.percentile(boot_f1, [2.5, 97.5])
print("=" * 70)
print("BALANCED ACCURACY + MACRO-F1 95% CI (bootstrap, n=2000 resamples)")
print("=" * 70)
print(f"  Balanced accuracy: {bal_acc:.4f}")
print(f"  Macro-F1: {f1_xgb:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print()

# ── Comparison vs. canonical Phase 8 flat RF baseline ──────────────────────
print("=" * 70)
print("COMPARISON — flat XGBoost vs. canonical flat RF baseline")
print("=" * 70)
print(f"  Flat RF  (Phase 8, canonical):  acc=0.9424  macro-F1=0.8159")
print(f"  Flat XGB (this script):         acc={acc_xgb:.4f}  macro-F1={f1_xgb:.4f}")
print()

# ── Feature importances ────────────────────────────────────────────────────
print("=" * 70)
print("FEATURE IMPORTANCES (top 20, gain-based)")
print("=" * 70)
imp = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat, val in imp.head(20).items():
    bar = "█" * int(val * 300)
    print(f"  {feat:<30}  {val:.4f}  {bar}")
print()

# ── Per-model breakdown ────────────────────────────────────────────────────
print("=" * 70)
print("PER-MODEL ACCURACY (test on seed 4)")
print("=" * 70)
for model in sorted(test_df["model"].unique()):
    sub = test_df[test_df["model"] == model]
    y_t = le.transform(sub["failure_mode"].values)
    y_p = xgb.predict(sub[FEATURE_COLS].values)
    acc = accuracy_score(y_t, y_p)
    f1v = f1_score(y_t, y_p, average="macro", zero_division=0)
    print(f"  {model:<25}  acc={acc:.4f}  macro-F1={f1v:.4f}  n={len(sub)}")
print()

# ── Save model ──────────────────────────────────────────────────────────────
out = {
    "model": xgb,
    "label_encoder": le,
    "feature_cols": FEATURE_COLS,
    "label_order": ALL_LABELS,
    "train_seeds": TRAIN_SEEDS,
    "test_seed": TEST_SEED,
}
pkl_path = os.path.join(_args.out_dir, "flat_xgboost_classifier.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(out, f)
print(f"[flat-xgb] Model saved -> {pkl_path}")
print("[flat-xgb] Done.")
