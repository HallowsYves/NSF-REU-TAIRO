"""
Episode runner for TAIRO FetchReach-v4 benchmarks.

Provides:
    EpisodeResult — dataclass holding per-episode summary metrics.
    run_episode   — runs one episode under a given method and attack condition.

Supported method strings
------------------------
    "rule_based"              Proportional reaching controller
    "sac" / "sac_her"         SB3 model (pass model= or policy_fn=)
    "recovery_aware_sac_her"  SB3 model with recovery damping enabled

Attack conditions (Week 5 additions marked with *)
---------------------------------------------------
    Observation-level:
        sensor_noise          Gaussian noise on all fields
        target_shift          Goal spoofing (immediate onset)
        sensor_dropout   *    Zeros obs["observation"] entirely
        sensor_bias      *    Constant per-dim offset on obs["observation"]
        goal_spoof_immediate* Goal shift from step 0
        goal_spoof_midep *    Goal shift from step 20 onward

    Action-level:
        action_noise, action_scale, action_reversal, action_delay

All attack functions, observation utilities, and recovery logic are
imported from their respective modules so this file contains only
orchestration logic.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import MAX_EPISODE_STEPS, SAFETY_ARM_JERK_THRESHOLD, SAFETY_GRIPPER_JERK_THRESHOLD
from envs.fetchreach_env import distance_to_goal
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack
from policies.rule_based_policy import rule_based_reach_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pnp_spatial_fields(
    obs: Dict[str, np.ndarray],
    goal_offset: Optional[np.ndarray],
) -> Dict:
    """Extract ground-truth spatial fields from a PickAndPlace obs dict.

    Returns NaN-filled entries for FetchReach (10-dim obs), so the step log
    schema is uniform across both environments.
    """
    obs_vec = obs["observation"]
    if len(obs_vec) >= 25:
        gripper_pos  = obs_vec[0:3]
        object_pos   = obs_vec[3:6]
        object_velp  = obs_vec[14:17]
        grip_velp    = obs_vec[20:23]
        gripper_aper = float(np.sum(obs_vec[9:11]))
        true_goal    = np.asarray(obs["desired_goal"], dtype=np.float64)
        perc_goal    = true_goal + goal_offset if goal_offset is not None else true_goal
        dist_to_obj  = float(np.linalg.norm(gripper_pos - object_pos))
        dist_to_tg   = float(np.linalg.norm(object_pos  - true_goal))
        dist_to_pg   = float(np.linalg.norm(object_pos  - perc_goal))
    else:
        gripper_pos  = object_pos  = object_velp = grip_velp = np.full(3, np.nan)
        gripper_aper = np.nan
        true_goal    = perc_goal   = np.full(3, np.nan)
        dist_to_obj  = dist_to_tg  = dist_to_pg = np.nan

    return {
        "object_pos_x": float(object_pos[0]),
        "object_pos_y": float(object_pos[1]),
        "object_pos_z": float(object_pos[2]),
        "gripper_pos_x": float(gripper_pos[0]),
        "gripper_pos_y": float(gripper_pos[1]),
        "gripper_pos_z": float(gripper_pos[2]),
        "gripper_aperture": gripper_aper,
        "object_velp_x": float(object_velp[0]),
        "object_velp_y": float(object_velp[1]),
        "object_velp_z": float(object_velp[2]),
        "grip_velp_x": float(grip_velp[0]),
        "grip_velp_y": float(grip_velp[1]),
        "grip_velp_z": float(grip_velp[2]),
        "true_goal_x": float(true_goal[0]),
        "true_goal_y": float(true_goal[1]),
        "true_goal_z": float(true_goal[2]),
        "perceived_goal_x": float(perc_goal[0]),
        "perceived_goal_y": float(perc_goal[1]),
        "perceived_goal_z": float(perc_goal[2]),
        "distance_to_object": dist_to_obj,
        "distance_to_true_goal": dist_to_tg,
        "distance_to_perceived_goal": dist_to_pg,
    }


def _action_smoothness(actions: List[np.ndarray]) -> float:
    """Mean step-to-step action-difference norm. Lower = smoother control."""
    if len(actions) < 2:
        return 0.0
    diffs = [np.linalg.norm(actions[i] - actions[i - 1]) for i in range(1, len(actions))]
    return float(np.mean(diffs))


def _action_magnitude(actions: List[np.ndarray]) -> float:
    """Mean action norm. Large values may indicate instability or attack amplification."""
    if not actions:
        return 0.0
    return float(np.mean([np.linalg.norm(a) for a in actions]))


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Per-episode summary record. One row in the episode-level CSV."""
    method: str
    condition: str
    seed: int
    attack_level: float
    total_reward: float
    success: float           # 1.0 if is_success is True at the final timestep
    final_distance: float    # distance_to_goal at last step
    episode_length: int
    action_smoothness: float
    action_magnitude: float
    safety_violation: float  # 1.0 if any step exceeded action norm threshold
    recovery_used: float     # 1.0 if recovery was triggered at any step
    first_success_step: float = float("nan")  # first timestep where is_success=1.0, else NaN


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,
    method: str,
    seed: int,
    episode_in_seed: int = 0,
    condition: str = "clean",
    attack_level: float = 0.0,
    model=None,
    policy_fn: Optional[Callable] = None,
    use_recovery: bool = False,
    recovery_version: str = "v3",
    target_shift_step: int = 25,
    max_steps: int = MAX_EPISODE_STEPS,
    recovery_v4_classifier: Optional[Dict] = None,
    recovery_v4_calibration: Optional[Dict] = None,
    recovery_v4_checkpoint: str = "clean_2M",
    level4_classifier: Optional[Dict] = None,
) -> Tuple[EpisodeResult, pd.DataFrame]:
    """Run one episode and return a summary plus a step-level log DataFrame.

    Policy resolution order
    -----------------------
    1. ``policy_fn`` if provided (any callable ``(env, obs) -> action``).
    2. ``method`` string dispatch:
       - ``"rule_based"``  → rule_based_reach_policy
       - ``"sac"``, ``"sac_her"``, ``"sac_plain"`` → model.predict
    3. Fallback: rule_based_reach_policy.

    Args:
        env:               Gymnasium environment (already created).
        method:            Policy identifier string.
        seed:              Outer benchmark seed (0-4). Stored verbatim in the
                            output rows so downstream per-seed grouping is unaffected.
        episode_in_seed:   Index of this episode within its (seed, condition, method)
                            block of N_EPISODES_PER_SEED episodes (0-29). Combined with
                            ``seed`` to build the env.reset() seed below (paper §IV.C:
                            reset_seed = 100*seed + episode_in_seed), so each of the 30
                            episodes in a seed block draws an independent initial
                            spawn/goal instead of all 30 sharing one.
        condition:         Attack condition name (see module docstring).
        attack_level:      Float parameter for the attack (e.g., noise_std).
        model:             Trained SB3 model (required for sac / sac_her methods).
        policy_fn:         Optional callable that overrides ``method`` dispatch.
        use_recovery:      If True, apply TAIRO C5 recovery damping each step.
        target_shift_step: Step at which legacy target_shift activates.
        recovery_v4_classifier: Required when method in {"sac_her_recovery_v4",
                            "sac_her_recovery_v4_hx"}. The loaded
                            results/classifier_seedfix/online_failure_classifier.pkl
                            dict ({'model', 'feature_cols', ...}). Loaded once by
                            the caller (mirrors how `model` is already loaded
                            once and reused across episodes) -- not reloaded
                            per episode. Both v4 and v4_hx share the same
                            classifier artifact -- v4_hx only adds a stage-
                            gating step on top of the same class probabilities
                            (see recovery/recovery_v4_hx.py).
        recovery_v4_calibration: Required when method in {"sac_her_recovery_v4",
                            "sac_her_recovery_v4_hx"}. The loaded
                            results/classifier_seedfix/recovery_v4_trigger_calibration.pkl
                            dict ({checkpoint_name: clean_pfail_p95}).
        recovery_v4_checkpoint: Which checkpoint's calibration entry to use
                            (default "clean_2M" -- Tier 1 CCAR is currently
                            scoped to clean_2M only; see RECOVERY_V4.md
                            section 2.6). Only meaningful when
                            method in {"sac_her_recovery_v4", "sac_her_recovery_v4_hx",
                            "sac_her_recovery_v4_hx2"}.
        level4_classifier: Required when method == "sac_her_recovery_v4_hx2".
                            The loaded results/classifier_level4/level4_classifier.pkl
                            dict ({'model', 'feature_cols', 'label_order', ...}).
                            Second classifier artifact, distinct from
                            recovery_v4_classifier -- see recovery/recovery_v4_hx2.py.

    Returns:
        Tuple of (EpisodeResult, step_log_DataFrame).
    """
    # Recovery version is encoded in the method name — derive it here.
    _use_recovery = method in {"sac_her_recovery_v2", "sac_her_recovery_v3",
                                "sac_her_recovery_v4", "sac_her_recovery_v4_hx",
                                "sac_her_recovery_v4_hx2"}
    _use_recovery_v4 = method in {"sac_her_recovery_v4", "sac_her_recovery_v4_hx",
                                   "sac_her_recovery_v4_hx2"}
    _use_recovery_v4_hx2 = method == "sac_her_recovery_v4_hx2"
    if _use_recovery_v4:
        from recovery.recovery_v4 import TriggerWeight, ExpertState, EPSILON as EPSILON_V4
        if method == "sac_her_recovery_v4_hx2":
            from recovery.recovery_v4_hx2 import recovery_step_hx2 as recovery_step
            if level4_classifier is None:
                raise ValueError(
                    "run_episode: method='sac_her_recovery_v4_hx2' requires "
                    "level4_classifier to be loaded once by the caller and "
                    "passed in, in addition to recovery_v4_classifier."
                )
        elif method == "sac_her_recovery_v4_hx":
            from recovery.recovery_v4_hx import recovery_step_hx as recovery_step
        else:
            from recovery.recovery_v4 import recovery_step
        if recovery_v4_classifier is None or recovery_v4_calibration is None:
            raise ValueError(
                f"run_episode: method='{method}' requires "
                "recovery_v4_classifier and recovery_v4_calibration to be "
                "loaded once by the caller and passed in (mirrors how "
                "`model` is loaded once, not per episode)."
            )
        if recovery_v4_checkpoint not in recovery_v4_calibration:
            raise ValueError(
                f"run_episode: recovery_v4_checkpoint='{recovery_v4_checkpoint}' "
                f"has no entry in recovery_v4_calibration "
                f"(available: {list(recovery_v4_calibration.keys())}). "
                "Tier 1 CCAR is currently scoped to clean_2M only -- see "
                "RECOVERY_V4.md section 2.6."
            )
        recovery_v4_trigger = TriggerWeight(
            clean_pfail_p95=recovery_v4_calibration[recovery_v4_checkpoint]
        )
        recovery_v4_expert_state = ExpertState()
        recovery_state = None
    elif _use_recovery:
        if method == "sac_her_recovery_v2":
            from recovery.recovery_v2 import maybe_apply_recovery, RecoveryState
        else:
            from recovery.recovery_v3 import maybe_apply_recovery, RecoveryState
        recovery_state = RecoveryState()
    else:
        recovery_state = None

    reset_seed = 100 * seed + episode_in_seed
    obs, info = env.reset(seed=reset_seed)

    total_reward = 0.0
    actions: List[np.ndarray] = []
    step_logs: List[Dict] = []
    previous_action: Optional[np.ndarray] = None   # delay buffer for action_delay
    prev_executed: Optional[np.ndarray] = None     # C4 jerk comparand — always last executed
    previous_obs: Optional[Dict] = None
    step_distances: List[float] = []
    any_recovery = False
    first_recovery_step: float = float("nan")
    first_success_step: float = float("nan")

    # Per-episode constants sampled once for attacks that require a fixed offset.
    # object_pose_offset is only used by PickAndPlace object_pose_spoof; None here.
    bias_vector: Optional[np.ndarray] = None
    goal_offset: Optional[np.ndarray] = None
    object_pose_offset: Optional[np.ndarray] = None

    for t in range(max_steps):
        # -- Observation-level attacks ----------------------------------------
        policy_obs, bias_vector, goal_offset, object_pose_offset = apply_sensor_attack(
            condition, obs, t, bias_vector, goal_offset,
            attack_level=attack_level, target_shift_step=target_shift_step,
            object_pose_offset=object_pose_offset,
        )

        # -- Policy action -------------------------------------------------------
        if policy_fn is not None:
            action = policy_fn(env, policy_obs)
        elif method in {"sac_her", "sac_her_recovery_v2", "sac_her_recovery_v3",
                        "sac_her_recovery_v4", "sac_her_recovery_v4_hx",
                        "sac_her_recovery_v4_hx2"} and model is not None:
            action, _ = model.predict(policy_obs, deterministic=True)
        else:
            raise ValueError(f"run_episode: unknown method '{method}' or model is None")

        intended_action = np.asarray(action, dtype=np.float32).copy()

        # -- Action-level attacks ------------------------------------------------
        executed_action = apply_action_attack(
            condition, intended_action, previous_action, attack_level=attack_level
        )

        # -- Recovery (TAIRO C5) -------------------------------------------------
        recovery_triggered = False
        recovery_v4_weight = float("nan")
        recovery_v4_ema_pfail = float("nan")
        if _use_recovery_v4:
            # step_history_df holds steps [0, t-1] only -- step t has not
            # happened yet, same causal discipline as evaluation/causal_features.py.
            step_history_df = pd.DataFrame(step_logs)
            _recovery_kwargs = dict(
                policy_action=executed_action,
                obs=obs,                      # raw unattacked obs — same as v2/v3
                step_history_df=step_history_df,
                classifier_artifact=recovery_v4_classifier,
                trigger=recovery_v4_trigger,
                expert_state=recovery_v4_expert_state,
                step=t,
            )
            if _use_recovery_v4_hx2:
                _recovery_kwargs["level4_classifier_artifact"] = level4_classifier
            executed_action, _v4_probs, recovery_v4_weight = recovery_step(**_recovery_kwargs)
            # trigger.ema_pfail is mutated in place by recovery_step's call to
            # trigger.update() -- reading it here gives exactly the value that
            # produced recovery_v4_weight this step (or the unset 0.0 initial
            # value at the step-0 guard, before update() is ever called).
            recovery_v4_ema_pfail = recovery_v4_trigger.ema_pfail
            executed_action = np.clip(executed_action, env.action_space.low, env.action_space.high)
            recovery_triggered = recovery_v4_weight >= EPSILON_V4
            if recovery_triggered:
                any_recovery = True
                if np.isnan(first_recovery_step):
                    first_recovery_step = float(t)
        elif _use_recovery:
            executed_action, recovery_triggered = maybe_apply_recovery(
                obs=obs,                      # raw unattacked obs — recovery steers to real goal
                action=executed_action,
                prev_obs=previous_obs,
                prev_action=previous_action,
                step_distances=step_distances,
                step=t,
                env=env,
                state=recovery_state,
            )
            if recovery_triggered:
                any_recovery = True
                if np.isnan(first_recovery_step):
                    first_recovery_step = float(t)

        # -- Per-channel split jerk metric for C4 safety scoring -----------------
        # Always compare consecutive *executed* actions — the actual command stream
        # sent to the robot regardless of which component (base policy, delay buffer,
        # or recovery controller) produced it.  prev_executed tracks this exclusively.
        #
        # NOTE: previous_action serves a separate purpose (action_delay buffer) and
        # intentionally stores intended_action for that condition; do NOT use it here.
        # Step 0: prev_executed is None → skip (no prior executed action available).
        if prev_executed is not None:
            _arm_jerk  = float(np.linalg.norm(executed_action[:3] - prev_executed[:3]))
            _grip_jerk = float(abs(executed_action[3] - prev_executed[3]))
            safety_violation_step = float(
                _arm_jerk  > SAFETY_ARM_JERK_THRESHOLD or
                _grip_jerk > SAFETY_GRIPPER_JERK_THRESHOLD
            )
        else:
            safety_violation_step = 0.0

        previous_obs = obs
        prev_executed = executed_action.copy()   # always the last executed action
        # For action_delay, store the policy's intended action so the next step
        # replays it as a genuine 1-step lag. Storing executed_action would
        # perpetuate zeros forever (the confirmed bug from Week 5).
        previous_action = (
            intended_action.copy() if condition == "action_delay" else executed_action.copy()
        )
        actions.append(executed_action.copy())

        # -- Environment step ----------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(executed_action)
        total_reward += float(reward)

        current_distance = distance_to_goal(obs)
        step_distances.append(current_distance)   # feed recovery trend detector
        is_success = float(info.get("is_success", 0.0))

        if is_success == 1.0 and np.isnan(first_success_step):
            first_success_step = float(t)

        spatial = _pnp_spatial_fields(obs, goal_offset)

        step_logs.append({
            "method": method,
            "condition": condition,
            "seed": seed,
            "attack_level": attack_level,
            "timestep": t,
            "reward": float(reward),
            "distance_to_goal": current_distance,
            "is_success": is_success,
            "action_norm": float(np.linalg.norm(executed_action)),
            "intended_action_norm": float(np.linalg.norm(intended_action)),
            "safety_violation": safety_violation_step,
            "recovery_triggered": float(recovery_triggered),
            "recovery_v4_weight": recovery_v4_weight,  # NaN for non-v4 methods
            "recovery_v4_ema_pfail": recovery_v4_ema_pfail,  # NaN for non-v4 methods
            **spatial,
        })

        if terminated or truncated:
            break

    step_df = pd.DataFrame(step_logs)
    step_df["steps_to_recovery"] = first_recovery_step

    result = EpisodeResult(
        method=method,
        condition=condition,
        seed=seed,
        attack_level=float(attack_level),
        total_reward=float(total_reward),
        success=float(step_df["is_success"].iloc[-1] if len(step_df) else 0.0),
        final_distance=float(step_df["distance_to_goal"].iloc[-1] if len(step_df) else float("nan")),
        episode_length=int(len(step_df)),
        action_smoothness=_action_smoothness(actions),
        action_magnitude=_action_magnitude(actions),
        safety_violation=float(step_df["safety_violation"].max() if len(step_df) else 0.0),
        recovery_used=float(any_recovery),
        first_success_step=first_success_step,
    )

    return result, step_df
