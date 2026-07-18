"""
Recovery v4 Tier 1 (CCAR) -- Phase 4 small-scale sanity validation.

Not a benchmark, not a T-score input. Order-of-tens episodes only, per the
Phase 4 scope agreed before running this. Checks, in order:

  A. No crashes, no NaN actions, across clean + two attack conditions.
  B. w stays near 0 on clean episodes (no false-triggering regression).
  C. Recovery visibly activates on at least one attacked condition.
  D. The 'object_pose_spoof' -> 'spoofed_goal' label fix (found during
     Phase 3 wiring) actually works end-to-end: relocalization_expert
     receives non-zero weight under object_pose_spoof and its raw action
     points toward the (median-filtered) object position, not garbage.
     Before the fix this expert was silently dead code (always weight 0),
     so this check would previously have passed for the wrong reason
     (never running at all) -- this script deliberately isolates and
     inspects the expert's own contribution, not just the blended output.

Scoped to the clean_2M PickAndPlace checkpoint only, per RECOVERY_V4.md
section 2.6.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from config import (
    MODEL_PATH_PICKANDPLACE_2M,
    MAX_EPISODE_STEPS_PICKANDPLACE,
    ATTACK_LEVELS,
)

# Recovery v4 calibration/classifier artifacts live under classifier_seedfix
# (seed-independence-fix data), not the default config.CLASSIFIER_DIR --
# same override this whole session has used for these artifacts.
CLASSIFIER_DIR = "results/classifier_seedfix"
from envs.fetchpickandplace_env import make_env
from evaluation.episode_runner import run_episode, _pnp_spatial_fields
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack
from evaluation.causal_features import build_causal_features_online
from envs.fetchreach_env import distance_to_goal
from recovery.recovery_v4 import (
    recovery_step, TriggerWeight, ExpertState, build_trusted_state, EPSILON, EXPERTS,
    DEFAULT_ALPHA,
)

CHECKPOINT = "clean_2M"
N_EPISODES_PART_A = 10
CONDITIONS_PART_A = ["clean", "sensor_dropout", "object_pose_spoof"]
N_EPISODES_PART_D = 5

print("[phase4] Loading clean_2M model and Recovery v4 artifacts ...")
with open(os.path.join(CLASSIFIER_DIR, "online_failure_classifier.pkl"), "rb") as f:
    classifier_artifact = pickle.load(f)
with open(os.path.join(CLASSIFIER_DIR, "recovery_v4_trigger_calibration.pkl"), "rb") as f:
    calibration = pickle.load(f)

env = make_env(seed=0)
model = SAC.load(MODEL_PATH_PICKANDPLACE_2M, env=env)

# ---------------------------------------------------------------------------
# Part A/B/C -- integration checks via the public run_episode() API
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART A/B/C: run_episode() integration checks (no crash/NaN, w near 0 on")
print("clean, recovery activates on attack)")
print(f"Active calibration: alpha={DEFAULT_ALPHA}, K_STEEPNESS={TriggerWeight.K_STEEPNESS}")
print("=" * 78)

overall_pass = True
for condition in CONDITIONS_PART_A:
    attack_level = ATTACK_LEVELS[condition]
    max_ws = []
    any_nan = False
    any_recovery = False
    crashed = False

    for ep in range(N_EPISODES_PART_A):
        try:
            result, step_df = run_episode(
                env=env,
                method="sac_her_recovery_v4",
                seed=0,
                episode_in_seed=ep,
                condition=condition,
                attack_level=attack_level,
                model=model,
                max_steps=MAX_EPISODE_STEPS_PICKANDPLACE,
                recovery_v4_classifier=classifier_artifact,
                recovery_v4_calibration=calibration,
                recovery_v4_checkpoint=CHECKPOINT,
            )
        except Exception as e:
            crashed = True
            print(f"  CRASH at condition={condition} ep={ep}: {e}")
            continue

        if step_df[["action_norm", "intended_action_norm"]].isna().any().any():
            any_nan = True
        w_series = step_df["recovery_v4_weight"].dropna()
        if len(w_series) > 0:
            max_ws.append(float(w_series.max()))
        if result.recovery_used > 0:
            any_recovery = True

    max_w_over_eps = max(max_ws) if max_ws else float("nan")
    mean_max_w = float(np.mean(max_ws)) if max_ws else float("nan")

    print(f"{condition:<22} crashed={crashed}  any_nan={any_nan}  "
          f"max_w={max_w_over_eps:.4f}  mean_max_w={mean_max_w:.4f}  "
          f"any_recovery={any_recovery}")

    if crashed or any_nan:
        overall_pass = False
    if condition == "clean" and (np.isnan(max_w_over_eps) or max_w_over_eps >= 0.10):
        print(f"  WARNING: clean-episode max w = {max_w_over_eps:.4f}, expected < 0.10")
        overall_pass = False
    if condition != "clean" and not any_recovery:
        print(f"  WARNING: no recovery activation observed under {condition}")

# ---------------------------------------------------------------------------
# Part D -- verify the spoofed_goal label fix end-to-end
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART D: object_pose_spoof -- verify relocalization_expert receives")
print("nonzero weight and produces a directionally sane action")
print("=" * 78)

condition = "object_pose_spoof"
attack_level = ATTACK_LEVELS[condition]

max_spoofed_goal_prob = 0.0
n_steps_with_reloc_weight = 0
n_direction_checks = 0
n_direction_sane = 0

for ep in range(N_EPISODES_PART_D):
    reset_seed = 100 * 0 + ep
    obs, info = env.reset(seed=reset_seed)
    trigger = TriggerWeight(clean_pfail_p95=calibration[CHECKPOINT])
    expert_state = ExpertState()
    step_logs = []
    previous_action = None
    bias_vector = None
    goal_offset = None
    object_pose_offset = None

    for t in range(MAX_EPISODE_STEPS_PICKANDPLACE):
        policy_obs, bias_vector, goal_offset, object_pose_offset = apply_sensor_attack(
            condition, obs, t, bias_vector, goal_offset,
            attack_level=attack_level, object_pose_offset=object_pose_offset,
        )
        action, _ = model.predict(policy_obs, deterministic=True)
        intended_action = np.asarray(action, dtype=np.float32).copy()
        executed_action = apply_action_attack(
            condition, intended_action, previous_action, attack_level=attack_level
        )

        step_history_df = pd.DataFrame(step_logs)
        final_action, class_probs, w = recovery_step(
            policy_action=executed_action,
            obs=obs,
            step_history_df=step_history_df,
            classifier_artifact=classifier_artifact,
            trigger=trigger,
            expert_state=expert_state,
            step=t,
        )
        final_action = np.clip(final_action, env.action_space.low, env.action_space.high)

        spoofed_prob = class_probs.get("spoofed_goal", 0.0)
        max_spoofed_goal_prob = max(max_spoofed_goal_prob, spoofed_prob)

        if spoofed_prob > EPSILON and len(step_history_df) > 0:
            n_steps_with_reloc_weight += 1
            # Isolate relocalization_expert's own raw action (not the blend)
            # to check its direction independent of the other four experts.
            feature_vec = build_causal_features_online(step_history_df)
            trusted_state = build_trusted_state(obs)
            reloc_action = EXPERTS["spoofed_goal"](trusted_state, feature_vec, expert_state)

            if np.any(np.isnan(reloc_action)):
                print(f"  ep={ep} t={t}: NaN in relocalization_expert action!")
            else:
                robust_pos = expert_state.median_object_pos(fallback=trusted_state["object_pos"])
                target_dir = robust_pos - trusted_state["gripper_pos"]
                if np.linalg.norm(target_dir) > 1e-4:
                    n_direction_checks += 1
                    cos_sim = float(
                        np.dot(reloc_action[:3], target_dir)
                        / (np.linalg.norm(reloc_action[:3]) * np.linalg.norm(target_dir) + 1e-8)
                    )
                    if cos_sim > 0.5:
                        n_direction_sane += 1

        previous_action = (
            intended_action.copy() if condition == "action_delay" else executed_action.copy()
        )

        obs, reward, terminated, truncated, info = env.step(final_action)
        current_distance = distance_to_goal(obs)
        is_success = float(info.get("is_success", 0.0))
        spatial = _pnp_spatial_fields(obs, goal_offset)
        step_logs.append({
            "method": "sac_her_recovery_v4", "condition": condition, "seed": 0,
            "attack_level": attack_level, "timestep": t, "reward": float(reward),
            "distance_to_goal": current_distance, "is_success": is_success,
            "action_norm": float(np.linalg.norm(final_action)),
            "intended_action_norm": float(np.linalg.norm(intended_action)),
            "safety_violation": 0.0, "recovery_triggered": float(w >= EPSILON),
            **spatial,
        })

        if terminated or truncated:
            break

print(f"max spoofed_goal probability observed: {max_spoofed_goal_prob:.4f}")
print(f"steps with spoofed_goal prob > EPSILON ({EPSILON}): {n_steps_with_reloc_weight}")
print(f"direction checks performed: {n_direction_checks}")
print(f"direction sane (cos_sim > 0.5 with median-filtered object direction): {n_direction_sane}")

fix_verified = n_steps_with_reloc_weight > 0 and n_direction_checks > 0 and (
    n_direction_sane / max(n_direction_checks, 1) > 0.5
)
print(f"\nspoofed_goal fix verified end-to-end: {fix_verified}")

print("\n" + "=" * 78)
print(f"OVERALL PART A/B/C PASS: {overall_pass}")
print(f"OVERALL PART D (fix verification) PASS: {fix_verified}")
print("=" * 78)

# ---------------------------------------------------------------------------
# Phase 5a -- larger live clean-episode batch for calibration refit
# ---------------------------------------------------------------------------
# Data collection only. Does not touch alpha/K_STEEPNESS. Uses the same
# run_episode() integration path as Part A, just more episodes and reading
# the per-step recovery_v4_ema_pfail column added alongside recovery_v4_weight
# for this purpose.
N_EPISODES_PHASE5A = 40

print("\n" + "=" * 78)
print(f"PHASE 5A/5C: live clean-episode batch ({N_EPISODES_PHASE5A} episodes, clean_2M, "
      f"active calibration alpha={DEFAULT_ALPHA}, K_STEEPNESS={TriggerWeight.K_STEEPNESS})")
print("=" * 78)

all_w = []
all_ema_pfail = []
per_episode_max_w = []
phase5a_crashed = False
phase5a_any_nan = False

for ep in range(N_EPISODES_PHASE5A):
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
        phase5a_crashed = True
        print(f"  CRASH at ep={ep}: {e}")
        continue

    if step_df[["action_norm", "intended_action_norm"]].isna().any().any():
        phase5a_any_nan = True

    w_series = step_df["recovery_v4_weight"].dropna()
    ema_series = step_df["recovery_v4_ema_pfail"].dropna()
    all_w.extend(w_series.tolist())
    all_ema_pfail.extend(ema_series.tolist())
    if len(w_series) > 0:
        per_episode_max_w.append(float(w_series.max()))

all_w = np.array(all_w)
all_ema_pfail = np.array(all_ema_pfail)
per_episode_max_w = np.array(per_episode_max_w)


def _pctiles(arr, label):
    print(f"  {label:<28} n={len(arr):<6} min={arr.min():.4f}  p50={np.percentile(arr,50):.4f}  "
          f"p95={np.percentile(arr,95):.4f}  p99={np.percentile(arr,99):.4f}  max={arr.max():.4f}")


print(f"crashed={phase5a_crashed}  any_nan={phase5a_any_nan}  "
      f"n_episodes_collected={len(per_episode_max_w)}\n")
_pctiles(all_w, "pooled step-level w")
_pctiles(all_ema_pfail, "pooled step-level ema_pfail")
_pctiles(per_episode_max_w, "per-episode max w")

print("\n[phase5a] Data collection complete. Stopping here for review before Phase 5b.")

env.close()
