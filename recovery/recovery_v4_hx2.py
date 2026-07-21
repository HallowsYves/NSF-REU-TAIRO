"""
Recovery v4-HX2: Level 4 (attack family) down-weighting, layered on top of
v4-HX's Level 1 stage-gated mixture.

Mentor-directed (2026-07-20). v4-HX (recovery/recovery_v4_hx.py) added
Level 1 task-stage conditioning to recovery_v4's expert mixture; evaluated
on the full clean_2M grid it showed NO measurable lift over plain
recovery_v4 (95% bootstrap CI on the success-rate delta straddled zero,
[-0.0158, +0.0079], 100/1650 discordant episode pairs split ~50/50). Per
the mentor's continued direction, this module adds the second planned
refinement anyway: Level 4 (attack family) prediction down-weights the
final blend weight when it confidently predicts `action_actuation`
(action_clipping/action_delay/action_reversal/grip_state_falsification).

Motivation for targeting `action_actuation` specifically: RECOVERY_V4.md
section 2.5 already argues action-channel attacks need secure actuation /
command authentication outside the compromised path, not better state
estimation -- i.e. state-based recovery experts have no real leverage
there, and blending them in may just be adding noise (or, per the v4/v4_hx
`grip_state_falsification` numbers -- sac_her 20.7%, v4 16.7%, v4_hx
16.0% -- may be actively harmful for that specific condition).

Requires a SECOND classifier artifact beyond recovery_v4's existing
online_failure_classifier.pkl: results/classifier_level4/level4_classifier.pkl
(saved for the first time this session by scripts/train_level4_classifier.py
-- no downstream consumer existed before this).

Additive only -- recovery_v4.py and recovery_v4_hx.py are both untouched.
New method variant `sac_her_recovery_v4_hx2`, not a replacement.
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


def recovery_step_hx2(policy_action: np.ndarray, obs: Dict, step_history_df,
                       classifier_artifact: Dict, level4_classifier_artifact: Dict,
                       trigger: TriggerWeight, expert_state: ExpertState, step: int):
    """One step of Recovery v4-HX2. Same signature as recovery_v4.recovery_step
    plus one extra required arg (level4_classifier_artifact), same call-site
    shape otherwise (drop-in for episode_runner.py alongside v4/v4_hx).
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

    # -- Level 4 down-weight -------------------------------------------------
    l4_model = level4_classifier_artifact["model"]
    l4_feature_cols = level4_classifier_artifact["feature_cols"]
    x_l4 = np.array([[feature_vec[c] for c in l4_feature_cols]])
    l4_probs = get_class_probs(l4_model, x_l4)
    l4_pred_class = max(l4_probs, key=l4_probs.get)
    l4_confidence = l4_probs[l4_pred_class]

    family_factor = 1.0
    if l4_pred_class == "action_actuation" and l4_confidence >= config.LEVEL4_CONFIDENT_THRESH:
        family_factor = config.LEVEL4_ACTION_ACTUATION_DOWNWEIGHT

    w_adjusted = w * family_factor
    final_action = (1 - w_adjusted) * policy_action + w_adjusted * recovery_action
    return final_action.astype(np.float32), class_probs, w_adjusted
