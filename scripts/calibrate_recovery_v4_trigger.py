"""
Recovery v4 Tier 1 -- per-checkpoint TriggerWeight calibration.

Computes clean_pfail_p95 for each trained model checkpoint (clean_2M,
clean_500k, randomized_2M, randomized_500k) from the existing causal
feature matrix. Same "multiplier x pooled-clean-max" calibration spirit as
the C4 jerk thresholds, applied per checkpoint rather than pooled globally
-- confirmed as the intended reading of RECOVERY_V4.md section 2.1/2.4
during Recovery v4 Tier 1 Phase 0.

No new rollout data is generated. The causal feature matrix already
contains clean-condition rows with intact 'condition' and 'model' columns
(excluded only from the classifier's training features, not dropped from
the file), so this script only filters and calls predict_proba.

Output
------
    results/classifier_seedfix/recovery_v4_trigger_calibration.pkl
        {model_name: clean_pfail_p95, ...}
    Console: per-checkpoint clean p_fail distribution (p50/p95/p99/max) and
        row counts, so the calibration is auditable.
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import CLASSIFIER_DIR as _DEFAULT_CLASSIFIER_DIR
from recovery.recovery_v4 import get_class_probs

# Same checkpoint list as scripts/build_online_classifier.py and
# scripts/build_causal_feature_matrix.py.
MODELS = ["clean_2M", "clean_500k", "randomized_2M", "randomized_500k"]

_parser = argparse.ArgumentParser()
_parser.add_argument("--classifier-dir", type=str, default=None)
_args = _parser.parse_args()
CLASSIFIER_DIR = _args.classifier_dir if _args.classifier_dir is not None else _DEFAULT_CLASSIFIER_DIR

print(f"[calibrate-v4] Loading causal feature matrix from {CLASSIFIER_DIR} ...")
feat_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "causal_feature_matrix.csv"))

with open(os.path.join(CLASSIFIER_DIR, "online_failure_classifier.pkl"), "rb") as f:
    clf_artifact = pickle.load(f)

model = clf_artifact["model"]
feature_cols = clf_artifact["feature_cols"]
success_idx = list(model.classes_).index("success")

clean_df = feat_df[feat_df["condition"] == "clean"].copy()
print(f"[calibrate-v4] {len(clean_df)} clean-condition rows across {clean_df['model'].nunique()} checkpoints\n")

calibration = {}
print(f"{'checkpoint':<18}{'n_rows':>8}{'p50':>10}{'p95':>10}{'p99':>10}{'max':>10}")
for model_name in MODELS:
    sub = clean_df[clean_df["model"] == model_name]
    if len(sub) == 0:
        print(f"[calibrate-v4] WARNING: no clean rows found for checkpoint '{model_name}', skipping")
        continue

    X = sub[feature_cols].values
    probs = model.predict_proba(X)
    p_success = probs[:, success_idx]
    p_fail = 1.0 - p_success

    p50 = float(np.percentile(p_fail, 50))
    p95 = float(np.percentile(p_fail, 95))
    p99 = float(np.percentile(p_fail, 99))
    pmax = float(p_fail.max())

    calibration[model_name] = p95
    print(f"{model_name:<18}{len(sub):>8}{p50:>10.4f}{p95:>10.4f}{p99:>10.4f}{pmax:>10.4f}")

# Spot-check the shared get_class_probs helper against the bulk computation
# above on one row, to confirm the two paths agree (classes_ ordering is
# the exact bug this helper exists to prevent).
_spot_row = clean_df[feature_cols].values[:1]
_spot_probs = get_class_probs(model, _spot_row)
_bulk_probs = model.predict_proba(_spot_row)[0]
_bulk_success = float(_bulk_probs[success_idx])
assert abs(_spot_probs["success"] - _bulk_success) < 1e-12, \
    "get_class_probs disagrees with bulk predict_proba indexing -- ordering bug"
print(f"\n[calibrate-v4] get_class_probs spot-check OK (success prob {_spot_probs['success']:.4f} matches bulk path)")

out_path = os.path.join(CLASSIFIER_DIR, "recovery_v4_trigger_calibration.pkl")
with open(out_path, "wb") as f:
    pickle.dump(calibration, f)
print(f"[calibrate-v4] Saved -> {out_path}")
print(f"[calibrate-v4] {calibration}")
