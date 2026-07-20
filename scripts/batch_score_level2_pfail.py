"""
Item 1 Phase 1a (B2) — Level 2 (anomaly detection), Candidate D batch scoring.

Runs the existing online/causal classifier (results/classifier_seedfix/
online_failure_classifier.pkl) predict_proba over every row of the causal
feature matrix (all sac_her episodes, all 4 models, all 11 conditions, all
14 per-episode checkpoints already present in causal_feature_matrix.csv) to
produce a p_fail = 1 - p_success column per (episode, checkpoint).

Mirrors the per-model p95 methodology of
scripts/calibrate_recovery_v4_trigger.py, extended here to report the full
percentile band (p50/p75/p95/p99) per model rather than only p95, and
additionally broken out per condition since Level 2 thresholds need to work
across attacked as well as clean rollouts (unlike the v4 trigger, which is
calibrated on clean-only rows).

Does NOT finalize or write any categorical Level 2 label. Output is
intermediate p_fail data only, for threshold review.

Output
------
    results/level2_pfail_scores.csv
        episode_idx, checkpoint_t, model, condition, seed, failure_mode, p_fail
    Console: per-model and per-model-per-condition percentile tables.
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import CLASSIFIER_DIR as _DEFAULT_CLASSIFIER_DIR

MODELS = ["clean_2M", "clean_500k", "randomized_2M", "randomized_500k"]

_parser = argparse.ArgumentParser()
_parser.add_argument("--classifier-dir", type=str, default=None,
                      help="Directory holding causal_feature_matrix.csv and "
                           "online_failure_classifier.pkl (default: CLASSIFIER_DIR "
                           "from config.py, i.e. results/classifier/ — pass "
                           "results/classifier_seedfix explicitly for the seed-fixed artifacts)")
_args = _parser.parse_args()
CLASSIFIER_DIR = _args.classifier_dir if _args.classifier_dir is not None else _DEFAULT_CLASSIFIER_DIR

print("[level2-pfail] Loading causal feature matrix and online classifier ...")
feat_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "causal_feature_matrix.csv"))

with open(os.path.join(CLASSIFIER_DIR, "online_failure_classifier.pkl"), "rb") as f:
    clf_artifact = pickle.load(f)

model = clf_artifact["model"]
feature_cols = clf_artifact["feature_cols"]
success_idx = list(model.classes_).index("success")
print(f"[level2-pfail] {len(feat_df)} rows, model.classes_={list(model.classes_)}\n")

X = feat_df[feature_cols].values
probs = model.predict_proba(X)
p_success = probs[:, success_idx]
p_fail = 1.0 - p_success

out_df = feat_df[["episode_idx", "checkpoint_t", "model", "condition", "seed", "failure_mode"]].copy()
out_df["p_fail"] = p_fail

out_path = "results/level2_pfail_scores.csv"
out_df.to_csv(out_path, index=False)
print(f"[level2-pfail] Saved -> {out_path}  ({len(out_df)} rows)\n")

# ── Per-model percentile bands (all conditions pooled) ─────────────────────
print("=" * 78)
print("PER-MODEL p_fail PERCENTILES (all 11 conditions pooled, all checkpoints)")
print("=" * 78)
print(f"{'model':<18}{'n_rows':>8}{'p50':>10}{'p75':>10}{'p95':>10}{'p99':>10}{'max':>10}")
for m in MODELS:
    sub = out_df[out_df["model"] == m]["p_fail"].values
    p50, p75, p95, p99 = np.percentile(sub, [50, 75, 95, 99])
    print(f"{m:<18}{len(sub):>8}{p50:>10.4f}{p75:>10.4f}{p95:>10.4f}{p99:>10.4f}{sub.max():>10.4f}")
print()

# ── Per-model, clean-only percentile bands (matches v4 trigger calibration scope) ──
print("=" * 78)
print("PER-MODEL p_fail PERCENTILES (clean condition only — matches v4 trigger calibration)")
print("=" * 78)
print(f"{'model':<18}{'n_rows':>8}{'p50':>10}{'p75':>10}{'p95':>10}{'p99':>10}{'max':>10}")
for m in MODELS:
    sub = out_df[(out_df["model"] == m) & (out_df["condition"] == "clean")]["p_fail"].values
    p50, p75, p95, p99 = np.percentile(sub, [50, 75, 95, 99])
    print(f"{m:<18}{len(sub):>8}{p50:>10.4f}{p75:>10.4f}{p95:>10.4f}{p99:>10.4f}{sub.max():>10.4f}")
print()

# ── Per-model, per-condition percentile bands ──────────────────────────────
print("=" * 78)
print("PER-MODEL x PER-CONDITION p_fail PERCENTILES")
print("=" * 78)
conditions = sorted(feat_df["condition"].unique())
for m in MODELS:
    print(f"\n-- {m} --")
    print(f"{'condition':<28}{'n_rows':>8}{'p50':>10}{'p75':>10}{'p95':>10}{'p99':>10}{'max':>10}")
    for c in conditions:
        sub = out_df[(out_df["model"] == m) & (out_df["condition"] == c)]["p_fail"].values
        if len(sub) == 0:
            continue
        p50, p75, p95, p99 = np.percentile(sub, [50, 75, 95, 99])
        print(f"{c:<28}{len(sub):>8}{p50:>10.4f}{p75:>10.4f}{p95:>10.4f}{p99:>10.4f}{sub.max():>10.4f}")

print("\n[level2-pfail] Done. NOTE: no categorical Level 2 thresholds were applied "
      "or saved — this script only produces the continuous p_fail column for review.")
