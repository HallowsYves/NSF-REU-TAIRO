"""
Retrain prep — Item 1 detection-delay evaluation, Tasks 1-2-4.

Runs the exact training procedure already established in
scripts/train_causal_classifier.py (Phase 9B) — same feature set, same
train/test split (seeds 0-3 train, seed 4 test), same RandomForest
hyperparameters (300 trees, class_weight="balanced", random_state=42) —
against results/classifier_seedfix/causal_feature_matrix.csv, and persists
the fitted model plus a per-checkpoint detection-delay breakdown.

train_causal_classifier.py itself is NOT modified — it is eval-only by
design (Phase B docstring: "Does NOT save a model yet"). This script does
not change that file; it independently re-runs the identical procedure and
adds the one new thing this task needs: persisting the artifact + a
per-checkpoint, per-class breakdown (Phase B only reports accuracy/macro-F1
per checkpoint, not per-class precision/recall).

Output
------
    results/classifier_causal_baseline/pooled_model.pkl
    results/classifier_causal_baseline/detection_delay_metrics.txt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                              f1_score, balanced_accuracy_score, precision_recall_fscore_support)

from evaluation.failure_mode_labeling import ALL_LABELS
from evaluation.causal_features import FORBIDDEN_FEATURES

CLASSIFIER_DIR = "results/classifier_seedfix"
OUT_DIR = "results/classifier_causal_baseline"
MODEL_OUT = os.path.join(OUT_DIR, "pooled_model.pkl")
METRICS_OUT = os.path.join(OUT_DIR, "detection_delay_metrics.txt")

TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]

lines = []
def log(s=""):
    print(s)
    lines.append(s)

# ── Load + split — identical to train_causal_classifier.py ──────────────
feat_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "causal_feature_matrix.csv"))

NON_FEATURE_COLS = {"episode_idx", "checkpoint_t", "failure_mode", "condition", "seed", "model",
                    *FORBIDDEN_FEATURES}
FEATURE_COLS = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]
assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES)

log(f"[train+save] Causal feature matrix: {len(feat_df)} rows, {len(FEATURE_COLS)} features")

train_df = feat_df[feat_df["seed"].isin(TRAIN_SEEDS)].copy()
test_df = feat_df[feat_df["seed"] == TEST_SEED].copy()

X_train, y_train = train_df[FEATURE_COLS].values, train_df["failure_mode"].values
X_test, y_test = test_df[FEATURE_COLS].values, test_df["failure_mode"].values

log(f"[train+save] Train: {len(X_train)} rows (seeds {TRAIN_SEEDS})")
log(f"[train+save] Test:  {len(X_test)} rows  (seed {TEST_SEED})")

# ── Train — identical hyperparameters to train_causal_classifier.py ─────
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
bal_acc = balanced_accuracy_score(y_test, y_pred)

# ── Task 1: pooled sanity check ──────────────────────────────────────────
log()
log("=" * 70)
log("TASK 1 — POOLED SANITY CHECK (all 14 checkpoints together)")
log("=" * 70)
log(f"  Accuracy:          {acc:.4f}   (reference: ~0.948)")
log(f"  Macro F1:          {macro_f1:.4f}   (reference: ~0.839)")
log(f"  Balanced accuracy: {bal_acc:.4f}")
log()

# ── Save model ────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out = {
    "model": rf,
    "feature_cols": FEATURE_COLS,
    "label_order": ALL_LABELS,
    "train_seeds": TRAIN_SEEDS,
    "test_seed": TEST_SEED,
    "checkpoint_policy": "trained on all 14 per-episode checkpoints pooled (t=19,29,...,149) "
                          "— same procedure as scripts/train_causal_classifier.py (Phase 9B)",
    "source_pipeline": "scripts/train_causal_classifier.py (Phase 9B), re-run unmodified "
                        "for artifact persistence — see scripts/train_and_save_causal_pooled.py",
}
with open(MODEL_OUT, "wb") as f:
    pickle.dump(out, f)
log(f"[train+save] Model saved -> {MODEL_OUT}")
log()

# ── Task 2: per-checkpoint detection-delay breakdown ─────────────────────
log("=" * 70)
log("TASK 2 — PER-CHECKPOINT DETECTION-DELAY BREAKDOWN")
log("=" * 70)

test_df = test_df.copy()
test_df["pred"] = y_pred

checkpoint_rows = []
for t, sub in test_df.groupby("checkpoint_t"):
    a = accuracy_score(sub["failure_mode"], sub["pred"])
    f = f1_score(sub["failure_mode"], sub["pred"], average="macro", zero_division=0)
    checkpoint_rows.append({"checkpoint_t": t, "accuracy": a, "macro_f1": f, "n": len(sub)})
    log(f"\n--- checkpoint_t = {t}  (n={len(sub)})  accuracy={a:.4f}  macro-F1={f:.4f} ---")
    prec, rec, f1c, sup = precision_recall_fscore_support(
        sub["failure_mode"], sub["pred"], labels=ALL_LABELS, zero_division=0
    )
    for lbl, p, r, f1v, s in zip(ALL_LABELS, prec, rec, f1c, sup):
        log(f"    {lbl:<28} precision={p:.4f}  recall={r:.4f}  f1={f1v:.4f}  support={s}")

checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values("checkpoint_t")
log()
log("Summary table (checkpoint_t, accuracy, macro_f1):")
log(checkpoint_df.to_string(index=False))

t19 = checkpoint_df[checkpoint_df["checkpoint_t"] == 19].iloc[0]
t149 = checkpoint_df[checkpoint_df["checkpoint_t"] == 149].iloc[0]
log()
log(f"[sanity] t=19  acc={t19['accuracy']:.4f} macro-F1={t19['macro_f1']:.4f}  (reference: ~0.917/0.765)")
log(f"[sanity] t=149 acc={t149['accuracy']:.4f} macro-F1={t149['macro_f1']:.4f}  (reference: ~0.961/0.881)")

# ── Task 4: false-alarm rate on clean episodes ────────────────────────────
log()
log("=" * 70)
log("TASK 4 — FALSE-ALARM RATE ON CLEAN EPISODES")
log("=" * 70)
log("Definition used: restricted to test rows where condition == 'clean' AND the "
    "hindsight episode label is 'success' (the genuinely nominal case — a clean "
    "episode that actually succeeded). False alarm = model predicts anything other "
    "than 'success' on such a row, i.e. flags a failure mode when nothing is wrong. "
    "Reported pooled and broken out by checkpoint_t, since a recovery trigger would "
    "query at every step, not just once per episode.")
log()

clean_success = test_df[(test_df["condition"] == "clean") & (test_df["failure_mode"] == "success")]
if len(clean_success) == 0:
    log("[task4] SKIPPED — no clean-condition rows with hindsight label 'success' found "
        "in the seed-4 test set; cannot compute without inventing a different definition.")
else:
    fa_rate = float((clean_success["pred"] != "success").mean())
    log(f"[task4] Clean+success test rows: {len(clean_success)}")
    log(f"[task4] Pooled false-alarm rate: {fa_rate:.4f}  ({(clean_success['pred'] != 'success').sum()}/{len(clean_success)})")
    log()
    log("By checkpoint_t:")
    for t, sub in clean_success.groupby("checkpoint_t"):
        fa_t = float((sub["pred"] != "success").mean())
        log(f"  t={t:>3}  n={len(sub):>4}  false_alarm_rate={fa_t:.4f}")

log()
log(f"[train+save] Metrics log saved -> {METRICS_OUT}")
with open(METRICS_OUT, "w") as f:
    f.write("\n".join(lines) + "\n")

# Stash checkpoint_df for the plotting script
checkpoint_df.to_csv(os.path.join(OUT_DIR, "_checkpoint_curve_data.csv"), index=False)
