"""
Retrain prep — Task 1: re-verify the existing post-hoc RF baseline.

Loads results/classifier_seedfix/feature_matrix.csv directly (does NOT
rebuild it from step logs) and reproduces the exact train/test split,
RandomForest hyperparameters, and evaluation from
scripts/train_failure_classifier.py, to confirm the cited 0.9424
accuracy / 0.8159 macro-F1 numbers.

Output: results/classifier_causal_baseline/posthoc_reverify_metrics.txt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, balanced_accuracy_score)

from evaluation.failure_mode_labeling import ALL_LABELS

FEATURE_MATRIX_PATH = "results/classifier_seedfix/feature_matrix.csv"
OUT_PATH = "results/classifier_causal_baseline/posthoc_reverify_metrics.txt"

TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]
FORBIDDEN_FEATURES = {"condition", "attack_level", "method"}
NON_FEATURE_COLS = {"episode_idx", "failure_mode", "condition", "seed", "model", *FORBIDDEN_FEATURES}

lines = []
def log(s=""):
    print(s)
    lines.append(s)

feat_df = pd.read_csv(FEATURE_MATRIX_PATH)
log(f"[reverify] Loaded {FEATURE_MATRIX_PATH}: {feat_df.shape}")

FEATURE_COLS = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]
assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES)
log(f"[reverify] {len(FEATURE_COLS)} feature columns")

train_df = feat_df[feat_df["seed"].isin(TRAIN_SEEDS)].copy()
test_df = feat_df[feat_df["seed"] == TEST_SEED].copy()
log(f"[reverify] Train: {len(train_df)} episodes (seeds {TRAIN_SEEDS})")
log(f"[reverify] Test:  {len(test_df)} episodes  (seed {TEST_SEED})")

X_train = train_df[FEATURE_COLS].values
y_train = train_df["failure_mode"].values
X_test = test_df[FEATURE_COLS].values
y_test = test_df["failure_mode"].values

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
mf1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
bal_acc = balanced_accuracy_score(y_test, y_pred)

log()
log("=" * 70)
log("RE-VERIFIED POST-HOC BASELINE (RF, feature_matrix.csv)")
log("=" * 70)
log(f"  Accuracy:          {acc:.4f}")
log(f"  Macro F1:          {mf1:.4f}")
log(f"  Balanced accuracy: {bal_acc:.4f}")
log(f"  Cited (CLAUDE.md): acc=0.9424, macro-F1=0.8159")
log()
log("Per-class report:")
log(classification_report(y_test, y_pred, labels=ALL_LABELS, zero_division=0, digits=4))
log("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred, labels=ALL_LABELS)
cm_df = pd.DataFrame(cm, index=[f"T:{l[:12]}" for l in ALL_LABELS],
                      columns=[f"P:{l[:12]}" for l in ALL_LABELS])
log(cm_df.to_string())

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\n[reverify] Log saved -> {OUT_PATH}")
