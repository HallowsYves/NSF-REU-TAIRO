"""
Recovery v4-HX6: Level-4-gated fast-attack trigger EMA, layered on top of
v4-HX2's Level 1 + Level 4 mixture.

Motivation (2026-07-22, mentor-directed continuation of the hx5 trigger-speed
work). hx5 (recovery/recovery_v4_hx5.py) sped up TriggerWeight's EMA by
RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER (4x) on the rising side, GLOBALLY --
for every condition, not just the goal-spoofing conditions it targeted.
Evaluated at full power (n=450): another genuine null on
goal_spoof_immediate/goal_spoof_midep (no confirmed lift over v4 or v4_hx2),
and a real, if unconfirmed, cost -- the previously-confirmed
grip_state_falsification win (v4_hx2 vs v4: +4.2pp, p=0.0034) dropped to a
non-BH-significant raw p=0.058 under hx5's global speed-up. The natural
explanation: speeding up the trigger's reaction to ANY rising p_fail also
changes its behavior on grip_state_falsification episodes, a condition hx2's
own Level 4 down-weight was specifically tuned to handle gently (see
recovery_v4_hx2.py's family_factor). hx5 never showed this was necessary --
grip_state_falsification is an action_actuation-family condition, not a
perception_state/goal_manipulation one, so there is no reason a fix aimed at
goal-spoof detection speed should touch it at all.

Fix: gate the fast-attack multiplier on Level 4's OWN attack-family
prediction, reusing the exact classifier call and confidence threshold
(config.LEVEL4_CONFIDENT_THRESH) v4-HX2's action_actuation down-weight
already uses -- just applied to the trigger's EMA speed instead of (in
addition to) the final blend weight. The fast alpha
(RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER, unchanged value -- hx5 already
validated 4x as a reasonable speed-up magnitude; what was untested was
whether it should apply everywhere) fires only when BOTH:
  (a) p_fail is rising (a new sustained failure, same as hx5's own gate), AND
  (b) Level 4 confidently predicts perception_state or goal_manipulation
      (config.ATTACK_FAMILY_MAP: object_pose_spoof, goal_spoof_immediate,
      goal_spoof_midep -- the conditions the ramp-lag investigation actually
      targeted).
Everywhere else (including action_actuation conditions like
grip_state_falsification, and sensor_info_loss conditions), the trigger keeps
v4-HX2's original, unmodified EMA speed -- so this change cannot touch the
grip_state_falsification pathway hx5 regressed, by construction rather than
by retuning.

Because the trigger's own speed now depends on Level 4's prediction, Level 4
classification must run BEFORE trigger.update() is called (unlike
v4-HX2/HX3/HX4/HX5, which only classify Level 4 after already knowing w will
be used, purely to down-weight it). The resulting l4_probs are reused for
the existing action_actuation down-weight afterward -- one classifier call
per step, not two.

Additive only -- recovery_v4.py, recovery_v4_hx.py, recovery_v4_hx2.py,
recovery_v4_hx3.py, recovery_v4_hx4.py, and recovery_v4_hx5.py are all
untouched. New method variant `sac_her_recovery_v4_hx6`, A/B-able against
the adopted `sac_her_recovery_v4_hx2` (and against hx5, since both modify
the same trigger for the same original motivation).
"""

import math
from typing import Dict

import numpy as np

import config
from recovery.recovery_v4 import (
    EPSILON,
    ExpertState,
    TriggerWeight,
    build_trusted_state,
    get_class_probs,
)
from recovery.recovery_v4_hx import compute_recovery_action_hx, compute_task_stage_online

_FAST_TRIGGER_FAMILIES = {"perception_state", "goal_manipulation"}


class GatedFastAttackTriggerWeight(TriggerWeight):
    """TriggerWeight with an asymmetric EMA, like hx5's FastAttackTriggerWeight,
    except the faster alpha only applies when Level 4 confidently predicts a
    perception_state/goal_manipulation attack family (see module docstring).
    `update()` therefore takes two extra args beyond the base class -- callers
    must pass the current step's Level 4 prediction, which recovery_step_hx6
    now computes before calling trigger.update() rather than after.
    """

    def update(self, p_success: float, l4_pred_class: str = None,
               l4_confidence: float = 0.0) -> float:
        p_fail = 1 - p_success
        rising = p_fail > self.ema_pfail
        family_matches = (
            l4_pred_class in _FAST_TRIGGER_FAMILIES
            and l4_confidence >= config.LEVEL4_CONFIDENT_THRESH
        )
        if rising and family_matches:
            effective_alpha = min(1.0, self.alpha * config.RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER)
        else:
            effective_alpha = self.alpha
        self.ema_pfail = effective_alpha * p_fail + (1 - effective_alpha) * self.ema_pfail
        w = 1 / (1 + math.exp(-self.steepness * (self.ema_pfail - self.midpoint)))
        return w


def recovery_step_hx6(policy_action: np.ndarray, obs: Dict, step_history_df,
                       classifier_artifact: Dict, level4_classifier_artifact: Dict,
                       trigger: GatedFastAttackTriggerWeight, expert_state: ExpertState, step: int):
    """One step of Recovery v4-HX6. Same signature/shape as
    recovery_v4_hx2.recovery_step_hx2 (drop-in for the same call site), except
    `trigger` must be a GatedFastAttackTriggerWeight instance (not a plain
    TriggerWeight) -- see episode_runner.py's dispatch wiring. Mixture is
    identical to v4-HX2's (compute_recovery_action_hx, unmodified expert
    wiring) -- only the trigger's EMA speed, and when it is gated on, differ.
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

    # Level 4 classification now runs BEFORE the trigger update -- the trigger's
    # own EMA speed depends on this prediction (see module docstring), unlike
    # v4-HX2/HX3/HX4/HX5 where Level 4 is only consulted after w is already
    # known to matter. Reused below for the action_actuation down-weight too.
    l4_model = level4_classifier_artifact["model"]
    l4_feature_cols = level4_classifier_artifact["feature_cols"]
    x_l4 = np.array([[feature_vec[c] for c in l4_feature_cols]])
    l4_probs = get_class_probs(l4_model, x_l4)
    l4_pred_class = max(l4_probs, key=l4_probs.get)
    l4_confidence = l4_probs[l4_pred_class]

    w = trigger.update(class_probs["success"], l4_pred_class, l4_confidence)

    trusted_state = build_trusted_state(obs)
    expert_state.update(trusted_state, feature_vec, step)

    if w < EPSILON:
        return policy_action, class_probs, w

    task_stage = compute_task_stage_online(feature_vec)
    recovery_action = compute_recovery_action_hx(
        trusted_state, class_probs, feature_vec, expert_state, task_stage)

    # -- Level 4 down-weight (unchanged from v4-HX2/HX3/HX4/HX5) -------------
    family_factor = 1.0
    if l4_pred_class == "action_actuation" and l4_confidence >= config.LEVEL4_CONFIDENT_THRESH:
        family_factor = config.LEVEL4_ACTION_ACTUATION_DOWNWEIGHT

    w_adjusted = w * family_factor
    final_action = (1 - w_adjusted) * policy_action + w_adjusted * recovery_action
    return final_action.astype(np.float32), class_probs, w_adjusted
