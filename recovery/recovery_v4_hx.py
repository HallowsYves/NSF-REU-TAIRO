"""
Recovery v4-HX: stage-gated extension of Recovery v4 Tier 1 CCAR.

Mentor-directed (2026-07-20, see PROJECT_CONTEXT.md and CLAUDE.md's
"Level 5 (Recovery Decision) — standing decision"): the TAIRO-HX classifier
hierarchy should actually improve recovery, not stay a separate offline
analysis track. `recovery/recovery_v4.py`'s CCAR conditions only on its own
online failure_mode classifier (Level-3-equivalent) -- it never uses
Level 1 (task stage) or Level 4 (attack family) at all. This module adds
Level 1 conditioning on top of the existing, unmodified CCAR mechanism.

Additive only -- `recovery_v4.py` is untouched. This is a NEW method
variant (`sac_her_recovery_v4_hx`), not a replacement, so it is directly
A/B-able against the already-evaluated `sac_her_recovery_v4` results
without risk to what's already paper-facing (CLAUDE.md section 14).

Design (confirmed via sign-off):
  - Level 1's task stage is computed ONLINE, for free -- it is a
    deterministic cascade over causal features already in `feature_vec`
    (`is_success_now`, `grasp_kinematic_ever_sofar`, `dttg_now`,
    `contact_streak_now`, `reached_ever_sofar`), the exact same cascade
    `scripts/build_level1_labels.py` uses offline. No new model, no new
    instrumentation.
  - Each of recovery_v4's 5 experts gets a stage-compatibility mask
    (`config.STAGE_EXPERT_COMPAT`): 1.0 in eligible stages, SOFT gating
    (`config.STAGE_EXPERT_SOFT_WEIGHT`, not a hard zero) elsewhere, so a
    Level-1 misfire cannot fully silence a genuinely-needed expert.
  - Everything else (trigger weight, trusted_state, ExpertState, the
    5 expert functions themselves) is reused unmodified from
    `recovery_v4.py` -- only the mixture step changes.
"""

from typing import Dict

import numpy as np

import config
from recovery.recovery_v4 import (
    ACTION_DIM,
    EPSILON,
    EXPERTS,
    ExpertState,
    TriggerWeight,
    build_trusted_state,
    get_class_probs,
)


def compute_task_stage_online(feature_vec: Dict) -> str:
    """Level 1 task stage, computed online from one step's causal features.

    Mirrors scripts/build_level1_labels.py's priority cascade exactly
    (most-advanced stage first), applied to a single live feature_vec
    instead of a batch DataFrame column. Same 5 causal features, already
    present in the feature_vec built by evaluation/causal_features.py's
    build_causal_features_online -- no new computation.
    """
    if feature_vec.get("is_success_now", 0.0) == 1:
        return "verifying_completion"
    if (feature_vec.get("grasp_kinematic_ever_sofar", 0.0) == 1
            and feature_vec.get("dttg_now", float("inf")) < config.LEVEL1_PLACE_RADIUS):
        return "placing"
    if feature_vec.get("grasp_kinematic_ever_sofar", 0.0) == 1:
        return "transporting"
    if feature_vec.get("contact_streak_now", 0) >= 1:
        return "grasping"
    if feature_vec.get("reached_ever_sofar", 0.0) == 1:
        return "aligning_gripper"
    return "approaching_object"


def compute_recovery_action_hx(trusted_state: Dict, class_probs: Dict[str, float],
                                feature_vec: Dict, expert_state: ExpertState,
                                task_stage: str) -> np.ndarray:
    """Stage-gated version of recovery_v4.compute_recovery_action.

    Same weighted-sum-of-experts mixture, but each expert's class-
    probability weight is additionally multiplied by a stage-compatibility
    factor (1.0 if `task_stage` is in that expert's eligible set,
    config.STAGE_EXPERT_SOFT_WEIGHT otherwise) before mixing.
    """
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    for cls, expert_fn in EXPERTS.items():
        weight = class_probs.get(cls, 0.0)
        if weight <= 0.0:
            continue
        eligible_stages = config.STAGE_EXPERT_COMPAT.get(cls, set(config.LEVEL1_STAGES))
        stage_factor = 1.0 if task_stage in eligible_stages else config.STAGE_EXPERT_SOFT_WEIGHT
        action += (weight * stage_factor) * expert_fn(trusted_state, feature_vec, expert_state)
    return action


def recovery_step_hx(policy_action: np.ndarray, obs: Dict, step_history_df,
                      classifier_artifact: Dict, trigger: TriggerWeight,
                      expert_state: ExpertState, step: int):
    """One step of Recovery v4-HX. Same signature/return shape as
    recovery_v4.recovery_step (drop-in for the same call site in
    episode_runner.py), except the mixture step calls
    compute_recovery_action_hx with the online Level 1 task stage.
    """
    policy_action = np.asarray(policy_action, dtype=np.float32)

    if step == 0 or len(step_history_df) == 0:
        return policy_action, {}, 0.0

    from evaluation.causal_features import build_causal_features_online

    feature_vec = build_causal_features_online(step_history_df)
    model = classifier_artifact["model"]
    feature_cols = classifier_artifact["feature_cols"]
    x = np.array([[feature_vec[c] for c in feature_cols]])
    class_probs = get_class_probs(model, x)

    w = trigger.update(class_probs["success"])

    trusted_state = build_trusted_state(obs)
    expert_state.update(trusted_state, feature_vec, step)

    if w < EPSILON:
        return policy_action, class_probs, w

    task_stage = compute_task_stage_online(feature_vec)
    recovery_action = compute_recovery_action_hx(
        trusted_state, class_probs, feature_vec, expert_state, task_stage)
    final_action = (1 - w) * policy_action + w * recovery_action
    return final_action.astype(np.float32), class_probs, w
