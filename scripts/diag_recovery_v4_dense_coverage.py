"""
Recovery v4 Tier 1 (CCAR) -- Phase B step 4: online-classifier coverage
re-check (dense vs sparse), clean_2M only.

Reuses the exact Phase 5a live clean-episode batch from
scripts/diag_recovery_v4_phase4_sanity.py: run N clean episodes through the
public run_episode() integration path with method="sac_her_recovery_v4",
read the per-step recovery_v4_weight / recovery_v4_ema_pfail columns, and
report pooled step-level w, pooled step-level ema_pfail, and per-episode
max-w percentiles (min/p50/p95/p99/max) -- the same table Phase 5a printed.

The ONLY thing that varies between runs is which online_failure_classifier.pkl
is loaded (--classifier-dir). The trigger math is held fixed: alpha and
K_STEEPNESS come from recovery/recovery_v4.py unchanged, and the midpoint
(clean_pfail_p95) comes from the SAME recovery_v4_trigger_calibration.pkl for
both runs (--calibration-dir, default results/classifier_seedfix). This
isolates the effect of classifier training-data density on clean-episode
blend weight, per the Phase B prompt ("about the classifier's training data
density, not the trigger math"). Re-deriving the midpoint from the dense
classifier's clean distribution would be a Phase 5-style recalibration and is
deliberately out of scope here.

Scoped to clean_2M, consistent with RECOVERY_V4.md section 2.6.
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC

from config import (
    MODEL_PATH_PICKANDPLACE_2M,
    MAX_EPISODE_STEPS_PICKANDPLACE,
    ATTACK_LEVELS,
)
from envs.fetchpickandplace_env import make_env
from evaluation.episode_runner import run_episode
from recovery.recovery_v4 import TriggerWeight, DEFAULT_ALPHA

CHECKPOINT = "clean_2M"

_parser = argparse.ArgumentParser()
_parser.add_argument("--classifier-dir", type=str, default="results/classifier_seedfix",
                     help="Dir holding online_failure_classifier.pkl to test.")
_parser.add_argument("--calibration-dir", type=str, default="results/classifier_seedfix",
                     help="Dir holding recovery_v4_trigger_calibration.pkl (held FIXED "
                          "across sparse/dense runs).")
_parser.add_argument("--n-episodes", type=int, default=40)
_parser.add_argument("--label", type=str, default="",
                     help="Free-text label printed in the header (e.g. 'SPARSE' / 'DENSE').")
_args = _parser.parse_args()

print(f"[coverage] classifier-dir = {_args.classifier_dir}")
with open(os.path.join(_args.classifier_dir, "online_failure_classifier.pkl"), "rb") as f:
    classifier_artifact = pickle.load(f)
with open(os.path.join(_args.calibration_dir, "recovery_v4_trigger_calibration.pkl"), "rb") as f:
    calibration = pickle.load(f)

print(f"[coverage] calibration-dir = {_args.calibration_dir}  midpoint(clean_2M)={calibration[CHECKPOINT]:.4f}")
print(f"[coverage] trigger math (fixed): alpha={DEFAULT_ALPHA}, K_STEEPNESS={TriggerWeight.K_STEEPNESS}")
print(f"[coverage] classifier checkpoint_policy: {classifier_artifact.get('checkpoint_policy', 'n/a')}")

env = make_env(seed=0)
model = SAC.load(MODEL_PATH_PICKANDPLACE_2M, env=env)

N = _args.n_episodes
print("\n" + "=" * 78)
print(f"PHASE B COVERAGE RE-CHECK {_args.label}: {N} live clean episodes (clean_2M)")
print("=" * 78)

all_w = []
all_ema_pfail = []
per_episode_max_w = []
crashed = False
any_nan = False
successes = 0

for ep in range(N):
    try:
        result, step_df = run_episode(
            env=env,
            method="sac_her_recovery_v4",
            seed=0,
            episode_in_seed=ep,
            condition="clean",
            attack_level=ATTACK_LEVELS["clean"],
            model=model,
            max_steps=MAX_EPISODE_STEPS_PICKANDPLACE,
            recovery_v4_classifier=classifier_artifact,
            recovery_v4_calibration=calibration,
            recovery_v4_checkpoint=CHECKPOINT,
        )
    except Exception as e:
        crashed = True
        print(f"  CRASH at ep={ep}: {e}")
        continue

    if step_df[["action_norm", "intended_action_norm"]].isna().any().any():
        any_nan = True

    w_series = step_df["recovery_v4_weight"].dropna()
    ema_series = step_df["recovery_v4_ema_pfail"].dropna()
    all_w.extend(w_series.tolist())
    all_ema_pfail.extend(ema_series.tolist())
    if len(w_series) > 0:
        per_episode_max_w.append(float(w_series.max()))
    successes += int(result.success > 0)

all_w = np.array(all_w)
all_ema_pfail = np.array(all_ema_pfail)
per_episode_max_w = np.array(per_episode_max_w)


def _pctiles(arr, label):
    print(f"  {label:<28} n={len(arr):<6} min={arr.min():.4f}  p50={np.percentile(arr,50):.4f}  "
          f"p95={np.percentile(arr,95):.4f}  p99={np.percentile(arr,99):.4f}  max={arr.max():.4f}")


print(f"crashed={crashed}  any_nan={any_nan}  n_episodes_collected={len(per_episode_max_w)}  "
      f"clean_success={successes}/{N}\n")
_pctiles(all_w, "pooled step-level w")
_pctiles(all_ema_pfail, "pooled step-level ema_pfail")
_pctiles(per_episode_max_w, "per-episode max w")

env.close()
print("\n[coverage] Done.")
