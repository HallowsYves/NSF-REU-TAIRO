"""
Retrain prep — false-alarm rate bootstrap CIs + early/mid/late binning.

Uses the already-trained pooled causal classifier
(results/classifier_causal_baseline/pooled_model.pkl, saved 2026-07-19) and
the existing causal_feature_matrix.csv test split (seed 4) to recompute
predictions on clean+success rows (n=420, the same population used for the
false-alarm rate reported in detection_delay_metrics.txt), then bootstrap
CIs around the per-checkpoint and binned false-alarm rates. No retraining,
no new episodes — reuses the saved model and existing data only.

Output
------
    results/classifier_causal_baseline/false_alarm_ci.txt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd

CLASSIFIER_DIR = "results/classifier_seedfix"
OUT_DIR = "results/classifier_causal_baseline"
MODEL_PATH = os.path.join(OUT_DIR, "pooled_model.pkl")
OUT_PATH = os.path.join(OUT_DIR, "false_alarm_ci.txt")

N_BOOT = 5000
RNG_SEED = 42

BINS = {
    "early (t=19-59)": [19, 29, 39, 49, 59],
    "mid (t=69-109)":  [69, 79, 89, 99, 109],
    "late (t=119-149)": [119, 129, 139, 149],
}

lines = []
def log(s=""):
    print(s)
    lines.append(s)

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
feature_cols = saved["feature_cols"]
test_seed = saved["test_seed"]

feat_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "causal_feature_matrix.csv"))
test_df = feat_df[feat_df["seed"] == test_seed].copy()
test_df["pred"] = model.predict(test_df[feature_cols].values)

clean_success = test_df[(test_df["condition"] == "clean") & (test_df["failure_mode"] == "success")].copy()
clean_success["false_alarm"] = (clean_success["pred"] != "success").astype(int)

log(f"[boot] Loaded pooled_model.pkl (train_seeds={saved['train_seeds']}, test_seed={test_seed})")
log(f"[boot] Clean+success test rows: {len(clean_success)}")

# Sanity check against detection_delay_metrics.txt point estimates
log()
log("=" * 70)
log("SANITY CHECK — pooled false-alarm rate vs. prior session's number")
log("=" * 70)
pooled_fa = clean_success["false_alarm"].mean()
log(f"  Pooled false-alarm rate: {pooled_fa:.4f}  (reference: 0.3095, 130/420)")
log()

rng = np.random.default_rng(RNG_SEED)

def bootstrap_ci(arr, n_boot=N_BOOT):
    arr = np.asarray(arr)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = arr[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return arr.mean(), lo, hi

# ── Task 1: per-checkpoint bootstrap CIs ─────────────────────────────────
log("=" * 70)
log(f"TASK 1 — PER-CHECKPOINT FALSE-ALARM RATE, 95% BOOTSTRAP CI (n_boot={N_BOOT})")
log("=" * 70)

checkpoint_rows = []
for t, sub in clean_success.groupby("checkpoint_t"):
    est, lo, hi = bootstrap_ci(sub["false_alarm"].values)
    checkpoint_rows.append({"checkpoint_t": t, "n": len(sub), "false_alarm_rate": est,
                             "ci_lo": lo, "ci_hi": hi})

ckpt_df = pd.DataFrame(checkpoint_rows).sort_values("checkpoint_t")
log()
log(f"{'checkpoint_t':>12}  {'n':>4}  {'rate':>8}  {'95% CI':>18}")
for _, r in ckpt_df.iterrows():
    log(f"{int(r['checkpoint_t']):>12}  {int(r['n']):>4}  {r['false_alarm_rate']:>8.4f}  "
        f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
log()

# ── Task 2: early/mid/late binning ────────────────────────────────────────
log("=" * 70)
log("TASK 2 — EARLY / MID / LATE BINNED FALSE-ALARM RATE, 95% BOOTSTRAP CI")
log("=" * 70)
log()

bin_rows = []
for bin_name, ts in BINS.items():
    sub = clean_success[clean_success["checkpoint_t"].isin(ts)]
    est, lo, hi = bootstrap_ci(sub["false_alarm"].values)
    bin_rows.append({"bin": bin_name, "n": len(sub), "false_alarm_rate": est,
                      "ci_lo": lo, "ci_hi": hi})
    log(f"  {bin_name:<20}  n={len(sub):>4}  rate={est:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

bin_df = pd.DataFrame(bin_rows)
log()

# ── Task 3: does the CI support the steep early->mid drop? ───────────────
log("=" * 70)
log("TASK 3 — DOES THE CI SUPPORT THE 'STEEP EARLY-TO-MID DROP' CONCLUSION?")
log("=" * 70)
early = bin_df[bin_df["bin"].str.startswith("early")].iloc[0]
mid = bin_df[bin_df["bin"].str.startswith("mid")].iloc[0]
late = bin_df[bin_df["bin"].str.startswith("late")].iloc[0]

gap_early_mid = early["ci_lo"] - mid["ci_hi"]
gap_mid_late = mid["ci_hi"] - late["ci_lo"]  # expect overlap or small gap, mid<late slightly

log(f"  Early bin CI: [{early['ci_lo']:.4f}, {early['ci_hi']:.4f}]  (n={early['n']})")
log(f"  Mid bin CI:   [{mid['ci_lo']:.4f}, {mid['ci_hi']:.4f}]  (n={mid['n']})")
log(f"  Late bin CI:  [{late['ci_lo']:.4f}, {late['ci_hi']:.4f}]  (n={late['n']})")
log()
if early["ci_lo"] > mid["ci_hi"]:
    log(f"  Early-vs-mid CIs DO NOT overlap (gap = {gap_early_mid:.4f}). The drop from "
        f"early to mid is supported at the 95% bootstrap level, even accounting for the "
        f"small per-bin sample sizes.")
else:
    log(f"  Early-vs-mid CIs OVERLAP. The point-estimate drop is large, but at n~150/bin "
        f"the 95% bootstrap CIs are not cleanly separated — treat the drop as suggestive, "
        f"not statistically confirmed at this sample size.")
if mid["ci_lo"] <= late["ci_hi"] and late["ci_lo"] <= mid["ci_hi"]:
    log(f"  Mid-vs-late CIs overlap substantially — no strong evidence of a further "
        f"drop/rise from mid to late; consistent with the plateau seen in the accuracy/"
        f"macro-F1 curve over the same range.")
log()
log("  NOTE (per-checkpoint, not per-bin): individual per-checkpoint CIs above are wide "
    "(n=30/checkpoint) — read those as indicative only. The binned CIs are the more "
    "defensible per-group numbers for citation.")
log()
log("  This is evidence relevant to task-stage recognition (Phase 1 Item 2) and to the "
    "sparse-vs-dense query-cadence question, but does NOT settle either — no operating-"
    "point checkpoint or cadence decision is made here, per project norm.")

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
log()
log(f"[boot] Saved -> {OUT_PATH}")
