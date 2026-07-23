"""
Recovery v4-HX5: fast-attack/slow-release trigger EMA, layered on top of
v4-HX2's Level 1 + Level 4 mixture.

Motivation (2026-07-22). hx4 (recovery/recovery_v4_hx4.py) tested and RULED
OUT the hypothesis that the wrong expert was consuming the spoofed_goal
signal -- full remap of spoofed_goal onto transport_expert produced a
genuine, well-powered null on goal_spoof_immediate/goal_spoof_midep (no
significant change vs. either plain v4 or v4-HX2). A separate background
investigation into the classifier's own reliability for the spoofed_goal
class also came back clean (recall 0.947, precision 0.782, negligible
confusion with divergent_transport, confident within ~1 checkpoint of real
attack onset on both conditions) -- ruling out "the classifier signal itself
is weak or late" too.

What hx4's own step logs (results/data_recovery_v4_hx4/step_logs_*.csv)
surfaced instead: recovery_v4.py's TriggerWeight.update() uses a single,
symmetric EMA (alpha=DEFAULT_ALPHA ~= 0.05, ~20-step time constant) to smooth
p_fail into ema_pfail before the sigmoid maps it to a blend weight w. Measured
directly from the hx4 sweep:
  - goal_spoof_immediate (attacked from step 0): w does not cross even the
    EPSILON=0.05 activation floor until step 16; at step 39 it is still only
    ~0.24 (76% uncorrected policy action).
  - goal_spoof_midep (onset at step 60): ~40 more steps after onset to reach
    w~=0.32.
recovery_v3's rule_based_reach_policy, by contrast, triggers within 3-10
steps and then applies a FULL, UNBLENDED override for a sustained window --
no gradual ramp at all. In a 150-step episode, losing the first 40-60+ steps
to a diluted (or fully absent) correction plausibly explains most of the
v3-vs-v4-family gap on goal-spoofing, independent of which expert eventually
fires once w does rise -- exactly what hx4's null result is consistent with.

Fix: an asymmetric ("fast attack, slow release") EMA on the TRIGGER only.
When p_fail is RISING (a new sustained failure), use a faster effective alpha
(config.RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER x the original) so the
trigger reacts on the order of v3's own trigger speed; when p_fail is FALLING
(recovering or naturally noisy-but-clean), keep the original, carefully-tuned
alpha unchanged -- so the existing clean-episode decay/false-positive
behavior (Phase 5b, RECOVERY_V4.md section 2.6) is not touched, only the
ramp-up speed on a genuinely new failure. This targets the ramp-lag
specifically, without re-opening the clean-episode calibration that the
midpoint/alpha/K constants already went through.

This is a different kind of fix than hx3/hx4 (which both changed which
expert consumes which class probability) -- this one changes the TRIGGER
that gates the whole mixture, upstream of the expert-selection question
entirely. Not combined with hx4's remap (which showed no effect) so the
trigger-speed fix's own effect stays cleanly attributable; built on top of
v4-HX2's mixture (Level 1 stage-gate + Level 4 action_actuation down-weight),
the currently adopted method, exactly as hx3/hx4 were.

Additive only -- recovery_v4.py, recovery_v4_hx.py, recovery_v4_hx2.py,
recovery_v4_hx3.py, and recovery_v4_hx4.py are all untouched. New method
variant `sac_her_recovery_v4_hx5`, A/B-able against the adopted
`sac_her_recovery_v4_hx2`.
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


class FastAttackTriggerWeight(TriggerWeight):
    """TriggerWeight with an asymmetric EMA: faster alpha while p_fail is
    rising (a new sustained failure), the original alpha while it is falling
    (recovering, or ordinary clean-episode noise). See module docstring.
    """

    def update(self, p_success: float) -> float:
        p_fail = 1 - p_success
        if p_fail > self.ema_pfail:
            effective_alpha = min(1.0, self.alpha * config.RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER)
        else:
            effective_alpha = self.alpha
        self.ema_pfail = effective_alpha * p_fail + (1 - effective_alpha) * self.ema_pfail
        w = 1 / (1 + math.exp(-self.steepness * (self.ema_pfail - self.midpoint)))
        return w


def recovery_step_hx5(policy_action: np.ndarray, obs: Dict, step_history_df,
                       classifier_artifact: Dict, level4_classifier_artifact: Dict,
                       trigger: FastAttackTriggerWeight, expert_state: ExpertState, step: int):
    """One step of Recovery v4-HX5. Same signature/shape as
    recovery_v4_hx2.recovery_step_hx2 (drop-in for the same call site), except
    `trigger` must be a FastAttackTriggerWeight instance (not a plain
    TriggerWeight) -- see episode_runner.py's dispatch wiring. Mixture is
    identical to v4-HX2's (compute_recovery_action_hx, unmodified expert
    wiring) -- only the trigger's EMA speed differs.
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

    # -- Level 4 down-weight (unchanged from v4-HX2/HX3/HX4) -----------------
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
