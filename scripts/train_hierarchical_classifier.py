"""
TAIRO-HX Item 1 -- hierarchical RF and hierarchical XGBoost, the two
still-missing legs of the four-way comparison (flat RF / flat XGBoost /
hierarchical RF / hierarchical XGBoost) described in TAIRO-HX.md Section 4
and PROJECT_CONTEXT.md's 2026-07-20 correction.

Episode-level, to be directly comparable to the canonical flat-RF baseline
(0.9424 acc / 0.8159 macro-F1, Phase 8, `feature_matrix.csv`) and the flat
XGBoost baseline (0.9485 / 0.8240, `scripts/train_flat_xgboost_classifier.py`)
-- NOT checkpoint-pooled, which PROJECT_CONTEXT.md explicitly flags as
reference material only, not a comparison-point substitute for Item 1.

Episode-level representation (confirmed 2026-07-20): each episode's final
checkpoint (t=149) row's base causal features, PLUS the out-of-fold chained
Level 1/2/3 outputs (one-hot predicted class + confidence scalar each) from
scripts/build_hierarchical_chain.py -- this is what makes it "hierarchical"
rather than flat. Target = failure_mode (same as the flat baselines), so
the comparison isolates the effect of the chain-conditioned feature
representation, holding the prediction target fixed.

Input:  results/hierarchical_chain_predictions.csv (from build_hierarchical_chain.py)
        results/classifier_seedfix/causal_feature_matrix.csv (base causal features)
Output: results/classifier_hierarchical/hierarchical_eval_summary.csv
        results/classifier_hierarchical/hierarchical_per_class_report_rf.csv
        results/classifier_hierarchical/hierarchical_per_class_report_xgb.csv
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              f1_score, balanced_accuracy_score)
from xgboost import XGBClassifier

import config
from evaluation.causal_features import FORBIDDEN_FEATURES
from evaluation.failure_mode_labeling import ALL_LABELS
from evaluation.oof_chaining import KEY_COLS, onehot_upstream

OUT_DIR = "results/classifier_hierarchical"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]
FINAL_CHECKPOINT_T = 149

LEVEL2_CLASSES = ["normal", "suspicious", "abnormal", "unknown"]

# ── Load episode-final-checkpoint rows only ──────────────────────────────
print("[hier-clf] Loading chain predictions + base causal features (t=149 only) ...")
chain_df = pd.read_csv("results/hierarchical_chain_predictions.csv")
chain_t149 = chain_df[chain_df["checkpoint_t"] == FINAL_CHECKPOINT_T].copy()

base_df = pd.read_csv("results/classifier_seedfix/causal_feature_matrix.csv")
base_t149 = base_df[base_df["checkpoint_t"] == FINAL_CHECKPOINT_T].copy()
NON_FEATURE_COLS = {"episode_idx", "checkpoint_t", "failure_mode", "condition",
                     "seed", "model", *FORBIDDEN_FEATURES}
BASE_FEATURE_COLS = [c for c in base_df.columns if c not in NON_FEATURE_COLS]

assert len(chain_t149) == len(base_t149) == 6600, \
    f"Expected 6600 episodes at t=149, got chain={len(chain_t149)} base={len(base_t149)}"

df = base_t149[KEY_COLS + ["failure_mode"] + BASE_FEATURE_COLS].merge(
    chain_t149[KEY_COLS + ["level1_pred_class", "level1_confidence",
                            "level2_pred_class", "level2_confidence",
                            "level3_pred_class", "level3_confidence"]],
    on=KEY_COLS, how="left")
assert len(df) == 6600
assert df["level1_pred_class"].isna().sum() == 0

# ── One-hot encode the chained upstream signals (same schema as the chain) ─
# onehot_upstream() returns KEY_COLS + f"{prefix}_confidence" + onehot cols;
# df already carries the confidence column from the merge above, so drop the
# duplicate before merging back in (else pandas would suffix it _x/_y).
l1_onehot = onehot_upstream(df, "level1", config.LEVEL1_STAGES).drop(columns=["level1_confidence"])
l2_onehot = onehot_upstream(df, "level2", LEVEL2_CLASSES).drop(columns=["level2_confidence"])
l3_onehot = onehot_upstream(df, "level3", ALL_LABELS).drop(columns=["level3_confidence"])

CHAIN_FEATURE_COLS = (
    [f"level1_{s}" for s in config.LEVEL1_STAGES] + ["level1_confidence"]
    + [f"level2_{c}" for c in LEVEL2_CLASSES] + ["level2_confidence"]
    + [f"level3_{l}" for l in ALL_LABELS] + ["level3_confidence"]
)

df = df.merge(l1_onehot, on=KEY_COLS, how="left")
df = df.merge(l2_onehot, on=KEY_COLS, how="left")
df = df.merge(l3_onehot, on=KEY_COLS, how="left")

FEATURE_COLS = BASE_FEATURE_COLS + CHAIN_FEATURE_COLS
print(f"[hier-clf] Episode-level rows: {len(df)}")
print(f"[hier-clf] Features: {len(BASE_FEATURE_COLS)} base causal + "
      f"{len(CHAIN_FEATURE_COLS)} chain-conditioned = {len(FEATURE_COLS)} total\n")

train_df = df[df["seed"].isin(TRAIN_SEEDS)].copy()
test_df = df[df["seed"] == TEST_SEED].copy()

train_ep = set(train_df["episode_idx"])
test_ep = set(test_df["episode_idx"])
overlap = train_ep & test_ep
assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} episode_idx overlap"
print(f"[hier-clf] Train: {len(train_df)} episodes  Test: {len(test_df)} episodes  "
      f"(episode_idx overlap = {len(overlap)}, expect 0)\n")

X_train = train_df[FEATURE_COLS].values
X_test = test_df[FEATURE_COLS].values
y_train_raw = train_df["failure_mode"].values
y_test_raw = test_df["failure_mode"].values

le = LabelEncoder()
le.fit(ALL_LABELS)
y_train = le.transform(y_train_raw)
y_test = le.transform(y_test_raw)

BASELINES = {
    "flat_rf": {"accuracy": 0.9424, "macro_f1": 0.8159},
    "flat_xgboost": {"accuracy": 0.9485, "macro_f1": 0.8240},
}


def _bootstrap_ci(y_true, y_pred, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    boot = []
    for _ in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        boot.append(f1_score(y_true_arr[idx], y_pred_arr[idx], average="macro", zero_division=0))
    return np.percentile(boot, [2.5, 97.5])


def _report(name, y_test, y_pred, model, feature_cols, out_csv_suffix):
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    ci_lo, ci_hi = _bootstrap_ci(y_test, y_pred)

    print("=" * 70)
    print(f"HIERARCHICAL {name.upper()}")
    print("=" * 70)
    print(f"  Accuracy: {acc:.4f}   Macro F1: {macro_f1:.4f}   Balanced acc: {bal_acc:.4f}")
    print(f"  Macro-F1 95% CI (bootstrap, n=2000): [{ci_lo:.4f}, {ci_hi:.4f}]")
    print()
    for base_name, base_vals in BASELINES.items():
        print(f"  vs. {base_name:<14} acc={base_vals['accuracy']:.4f} "
              f"(delta={acc - base_vals['accuracy']:+.4f})   "
              f"macro-F1={base_vals['macro_f1']:.4f} "
              f"(delta={macro_f1 - base_vals['macro_f1']:+.4f})")
    print()

    y_test_lab = le.inverse_transform(y_test)
    y_pred_lab = le.inverse_transform(y_pred)
    print("Per-class report:")
    report_str = classification_report(y_test_lab, y_pred_lab, labels=ALL_LABELS,
                                        zero_division=0, digits=3)
    print(report_str)
    report_dict = classification_report(y_test_lab, y_pred_lab, labels=ALL_LABELS,
                                         zero_division=0, digits=3, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(OUT_DIR, f"hierarchical_per_class_report_{out_csv_suffix}.csv"))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test_lab, y_pred_lab, labels=ALL_LABELS)
    cm_df = pd.DataFrame(cm, index=[f"T:{l[:16]}" for l in ALL_LABELS],
                                  columns=[f"P:{l[:16]}" for l in ALL_LABELS])
    print(cm_df.to_string())
    print()

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 15 feature importances (chain-conditioned features marked *):")
    for feat, val in imp.head(15).items():
        marker = "*" if feat in CHAIN_FEATURE_COLS else " "
        bar = "█" * int(val * 300)
        print(f"  {marker} {feat:<30}  {val:.4f}  {bar}")
    print()

    return {"accuracy": acc, "macro_f1": macro_f1, "balanced_accuracy": bal_acc,
            "macro_f1_ci_lo": ci_lo, "macro_f1_ci_hi": ci_hi}


# ── Hierarchical RF ────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train_raw)
y_pred_rf = le.transform(rf.predict(X_test))
rf_results = _report("RF", y_test, y_pred_rf, rf, FEATURE_COLS, "rf")

# ── Hierarchical XGBoost ───────────────────────────────────────────────────
class_counts = pd.Series(y_train).value_counts()
n_classes = len(class_counts)
n_samples = len(y_train)
class_weight_map = {c: n_samples / (n_classes * cnt) for c, cnt in class_counts.items()}
sample_weight = np.array([class_weight_map[c] for c in y_train])

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    objective="multi:softprob", num_class=len(ALL_LABELS),
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
xgb.fit(X_train, y_train, sample_weight=sample_weight)
y_pred_xgb = xgb.predict(X_test)
xgb_results = _report("XGBoost", y_test, y_pred_xgb, xgb, FEATURE_COLS, "xgb")

# ── Save summary ────────────────────────────────────────────────────────────
summary_rows = []
for name, res, base_name in [("hierarchical_rf", rf_results, "flat_rf"),
                              ("hierarchical_xgboost", xgb_results, "flat_xgboost")]:
    row = {"model": name, "n_train": len(X_train), "n_test": len(X_test),
           "n_features_base": len(BASE_FEATURE_COLS), "n_features_chain": len(CHAIN_FEATURE_COLS)}
    row.update(res)
    row[f"vs_{base_name}_accuracy_delta"] = res["accuracy"] - BASELINES[base_name]["accuracy"]
    row[f"vs_{base_name}_macro_f1_delta"] = res["macro_f1"] - BASELINES[base_name]["macro_f1"]
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUT_DIR, "hierarchical_eval_summary.csv"), index=False)

print("=" * 70)
print("FOUR-WAY COMPARISON SUMMARY (Item 1)")
print("=" * 70)
print(f"  Flat RF              (Phase 8, canonical):        acc=0.9424  macro-F1=0.8159")
print(f"  Flat XGBoost         (train_flat_xgboost_classifier.py): acc=0.9485  macro-F1=0.8240")
print(f"  Hierarchical RF      (this script):                acc={rf_results['accuracy']:.4f}  "
      f"macro-F1={rf_results['macro_f1']:.4f}")
print(f"  Hierarchical XGBoost (this script):                acc={xgb_results['accuracy']:.4f}  "
      f"macro-F1={xgb_results['macro_f1']:.4f}")
print(f"\n[hier-clf] Saved summary -> {os.path.join(OUT_DIR, 'hierarchical_eval_summary.csv')}")
print("[hier-clf] Done. No model saved (no downstream consumer yet).")
