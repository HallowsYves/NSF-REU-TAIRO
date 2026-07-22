"""
Recovery v4-HX3: re-gate relocalization_expert on Level 4's perception_state
signal, layered on top of v4-HX2's Level 1 + Level 4 mixture.

Motivation (2026-07-21). recovery_v4.py's relocalization_expert -- the one
expert built specifically for object-position corruption (median-filtered
robust object-position estimation) -- is weighted only by
class_probs["spoofed_goal"], because the online failure-mode classifier has
no dedicated object_pose_spoof label. Root-caused last session
(findings.md Phase 11 item 3): object_pose_spoof episodes almost never get
classified as spoofed_goal in practice (they land as
never_reached_object / reached_but_failed_grasp / divergent_transport
instead), so relocalization_expert's weight is already near-zero on the
exact condition it exists for -- a routing bug, not a capability gap.

v4-HX2 already computes a second, independent classifier's output every
step -- Level 4 (attack family) -- which has a perception_state class
specifically backed by object_pose_spoof. It was previously only used for
one thing (the action_actuation down-weight). This module reuses that
already-computed signal to also boost relocalization_expert's weight
whenever Level 4 is confident the corruption is perception/state-related,
instead of relying solely on the near-inert spoofed_goal proxy.

Combination rule: plain max(spoofed_goal, perception_state), not a
weighted blend -- confirmed via sign-off 2026-07-21. The codebase already
has three provisional, uncalibrated constants (STAGE_EXPERT_SOFT_WEIGHT,
LEVEL4_ACTION_ACTUATION_DOWNWEIGHT, LEVEL4_CONFIDENT_THRESH); a plain max
needs no new constant and reads as "trust whichever signal is more
confident," appropriate for a bounded fix that isn't meant to open a new
calibration project.

Additive only -- recovery_v4.py, recovery_v4_hx.py, and recovery_v4_hx2.py
are all untouched. New method variant `sac_her_recovery_v4_hx3`, A/B-able
against the already-adopted `sac_her_recovery_v4_hx2` without risk to it.
"""

from typing import Dict

import numpy as np

import config
from recovery.recovery_v4 import (
    ACTION_DIM,
    EPSILON,
    ExpertState,
    TriggerWeight,
    build_trusted_state,
    get_class_probs,
)
from recovery.recovery_v4_hx import compute_recovery_action_hx, compute_task_stage_online


def recovery_step_hx3(policy_action: np.ndarray, obs: Dict, step_history_df,
                       classifier_artifact: Dict, level4_classifier_artifact: Dict,
                       trigger: TriggerWeight, expert_state: ExpertState, step: int):
    """One step of Recovery v4-HX3. Same signature as recovery_v4_hx2.recovery_step_hx2
    (drop-in for the same call site in episode_runner.py).
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

    # -- Level 4 probabilities, computed before the mixture so relocalization_expert's
    # gating can use l4_probs["perception_state"] (moved earlier than v4-HX2's
    # equivalent block, which only needed it after the mixture for the down-weight).
    l4_model = level4_classifier_artifact["model"]
    l4_feature_cols = level4_classifier_artifact["feature_cols"]
    x_l4 = np.array([[feature_vec[c] for c in l4_feature_cols]])
    l4_probs = get_class_probs(l4_model, x_l4)
    l4_pred_class = max(l4_probs, key=l4_probs.get)
    l4_confidence = l4_probs[l4_pred_class]

    task_stage = compute_task_stage_online(feature_vec)

    # -- relocalization_expert re-gate: trust whichever signal is more confident that
    # this is object-position corruption, the spoofed_goal proxy or Level 4's dedicated
    # perception_state class.
    adjusted_class_probs = dict(class_probs)
    adjusted_class_probs["spoofed_goal"] = max(
        class_probs.get("spoofed_goal", 0.0), l4_probs.get("perception_state", 0.0)
    )
    recovery_action = compute_recovery_action_hx(
        trusted_state, adjusted_class_probs, feature_vec, expert_state, task_stage)

    # -- Level 4 down-weight (unchanged from v4-HX2, order-independent) ---------------
    family_factor = 1.0
    if l4_pred_class == "action_actuation" and l4_confidence >= config.LEVEL4_CONFIDENT_THRESH:
        family_factor = config.LEVEL4_ACTION_ACTUATION_DOWNWEIGHT

    w_adjusted = w * family_factor
    final_action = (1 - w_adjusted) * policy_action + w_adjusted * recovery_action
    return final_action.astype(np.float32), class_probs, w_adjusted
