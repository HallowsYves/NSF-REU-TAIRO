"""
Recovery v4-HX4: full remap of the spoofed_goal / relocalization_expert
wiring, layered on top of v4-HX2's Level 1 + Level 4 mixture.

Motivation (2026-07-22). Investigating why v4-HX/v4-HX2 underperform the much
simpler recovery_v3 on goal_spoof_immediate/goal_spoof_midep (v3: +2.9pp/+4.9pp
over sac_her at n=150; v4-HX2 essentially flat at n=450 -- see
PROJECT_CONTEXT.md's 2026-07-22 Next Steps item 2) surfaced a mechanistic
mismatch in recovery_v4.py's EXPERTS mapping: class_probs["spoofed_goal"]
drives relocalization_expert, but relocalization_expert only re-estimates the
OBJECT's own position (a median-filtered PD controller that never reads
trusted_state["true_goal"] at all) -- it was built for object_pose_spoof
(object-position corruption), per its own docstring, and has zero corrective
mechanism for a corrupted GOAL. Meanwhile transport_expert -- which DOES
PD-control toward trusted_state["true_goal"] -- is wired only to
divergent_transport, a broad catch-all, not to spoofed_goal.

recovery_v3's rule_based_reach_policy, by contrast, always computes
error = desired_goal - achieved_goal from the raw (unspoofed) obs the instant
any of its 3 generic triggers fire -- it needs no failure-mode classification
at all to correctly steer at the true goal. This plausibly explains why the
simpler v3 controller holds up better on goal-spoofing than v4's classifier-
mixture design: v4's mixture can classify the failure correctly as
spoofed_goal and STILL apply an expert with no pull toward the true goal.

Full remap (confirmed via sign-off 2026-07-22, chosen over a more
conservative additive-only alternative that would have left
relocalization_expert's spoofed_goal binding in place):
  - transport_expert's weight becomes
    class_probs["divergent_transport"] + class_probs["spoofed_goal"]
    (spoofed_goal moved OFF relocalization_expert, ONTO transport_expert --
    the mechanistically correct target, since spoofed_goal's own labeling
    precondition -- evaluation/failure_mode_labeling.py checks it only after
    a confirmed grasp -- matches transport_expert's own "object already
    grasped" precondition exactly).
  - relocalization_expert's weight becomes l4_probs["perception_state"]
    (Level 4's attack-family class, backed by object_pose_spoof; precision
    0.974 per results/classifier_level4/level4_per_class_report.csv for
    goal_manipulation, 0.635 for perception_state -- still far more targeted
    than the near-inert spoofed_goal signal hx3 diagnosed) instead of
    class_probs["spoofed_goal"] -- matches hx3's own root-cause finding that
    object_pose_spoof episodes almost never get classified as spoofed_goal,
    so perception_state is the correct, dedicated trigger for this expert
    rather than sharing a signal with the now fully-decoupled spoofed_goal
    class.
  - Both redirected weights keep their ORIGINAL stage-compatibility masks:
    transport_expert's added spoofed_goal contribution uses
    config.STAGE_EXPERT_COMPAT["divergent_transport"] (transporting/placing/
    verifying_completion) since transport_expert's own precondition --
    object already grasped -- doesn't change based on which class triggered
    it; relocalization_expert keeps config.STAGE_EXPERT_COMPAT["spoofed_goal"]
    (all stages eligible) since object/goal-position corruption isn't
    stage-bound regardless of which upstream classifier flags it.
  - The other three experts (grasp_stabilize_expert, regrasp_expert,
    approach_expert) are unchanged from v4-HX2 -- this module only touches
    the spoofed_goal / divergent_transport / perception_state triple.

This is a bigger architectural change than v4-HX3's plain max() addition
(that one added an OR-signal without removing anything; this one moves a
class-probability's destination expert entirely, decoupling spoofed_goal
and object_pose_spoof detection onto two different downstream experts
instead of the two sharing relocalization_expert) -- flagged and confirmed
via sign-off before implementation.

Additive only at the file level -- recovery_v4.py, recovery_v4_hx.py,
recovery_v4_hx2.py, and recovery_v4_hx3.py are all untouched. New method
variant `sac_her_recovery_v4_hx4`, A/B-able against the adopted
`sac_her_recovery_v4_hx2` (and independently against `sac_her_recovery_v4_hx3`,
which targeted a different, already-closed null result on object_pose_spoof
and is not combined with this fix).

Requires the same two classifier artifacts as v4-HX2/v4-HX3
(online_failure_classifier.pkl + level4_classifier.pkl) -- no new artifacts.
"""

from typing import Dict

import numpy as np

import config
from recovery.recovery_v4 import (
    ACTION_DIM,
    EPSILON,
    ExpertState,
    TriggerWeight,
    approach_expert,
    build_trusted_state,
    get_class_probs,
    grasp_stabilize_expert,
    regrasp_expert,
    relocalization_expert,
    transport_expert,
)
from recovery.recovery_v4_hx import compute_task_stage_online

# Experts whose class_probs binding is UNCHANGED from recovery_v4.py's
# original EXPERTS dict -- only spoofed_goal/divergent_transport/
# perception_state are remapped (see module docstring).
_UNCHANGED_EXPERTS = (
    ("reached_but_failed_grasp", grasp_stabilize_expert),
    ("grasped_but_dropped",      regrasp_expert),
    ("never_reached_object",     approach_expert),
)


def compute_recovery_action_hx4(trusted_state: Dict, class_probs: Dict[str, float],
                                 l4_probs: Dict[str, float], feature_vec: Dict,
                                 expert_state: ExpertState, task_stage: str) -> np.ndarray:
    """v4-HX4's remapped mixture -- see module docstring for the full
    rationale. Stage-gated exactly like recovery_v4_hx.compute_recovery_action_hx,
    just with transport_expert/relocalization_expert's weight sources swapped.
    """
    action = np.zeros(ACTION_DIM, dtype=np.float32)

    def _stage_factor(eligible_stages: set) -> float:
        return 1.0 if task_stage in eligible_stages else config.STAGE_EXPERT_SOFT_WEIGHT

    # -- transport_expert: divergent_transport + spoofed_goal (moved here) ---
    transport_weight = (
        class_probs.get("divergent_transport", 0.0) + class_probs.get("spoofed_goal", 0.0)
    )
    if transport_weight > 0.0:
        stage_factor = _stage_factor(config.STAGE_EXPERT_COMPAT["divergent_transport"])
        action += (transport_weight * stage_factor) * transport_expert(
            trusted_state, feature_vec, expert_state)

    # -- relocalization_expert: Level 4's perception_state only (not spoofed_goal) --
    reloc_weight = l4_probs.get("perception_state", 0.0)
    if reloc_weight > 0.0:
        stage_factor = _stage_factor(config.STAGE_EXPERT_COMPAT["spoofed_goal"])
        action += (reloc_weight * stage_factor) * relocalization_expert(
            trusted_state, feature_vec, expert_state)

    # -- Unchanged experts -----------------------------------------------------
    for cls, expert_fn in _UNCHANGED_EXPERTS:
        weight = class_probs.get(cls, 0.0)
        if weight <= 0.0:
            continue
        stage_factor = _stage_factor(config.STAGE_EXPERT_COMPAT[cls])
        action += (weight * stage_factor) * expert_fn(trusted_state, feature_vec, expert_state)

    return action


def recovery_step_hx4(policy_action: np.ndarray, obs: Dict, step_history_df,
                       classifier_artifact: Dict, level4_classifier_artifact: Dict,
                       trigger: TriggerWeight, expert_state: ExpertState, step: int):
    """One step of Recovery v4-HX4. Same signature as recovery_v4_hx2.recovery_step_hx2
    / recovery_v4_hx3.recovery_step_hx3 (drop-in for the same call site in
    episode_runner.py).
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

    l4_model = level4_classifier_artifact["model"]
    l4_feature_cols = level4_classifier_artifact["feature_cols"]
    x_l4 = np.array([[feature_vec[c] for c in l4_feature_cols]])
    l4_probs = get_class_probs(l4_model, x_l4)
    l4_pred_class = max(l4_probs, key=l4_probs.get)
    l4_confidence = l4_probs[l4_pred_class]

    task_stage = compute_task_stage_online(feature_vec)
    recovery_action = compute_recovery_action_hx4(
        trusted_state, class_probs, l4_probs, feature_vec, expert_state, task_stage)

    # -- Level 4 down-weight (unchanged from v4-HX2/HX3, order-independent) --
    family_factor = 1.0
    if l4_pred_class == "action_actuation" and l4_confidence >= config.LEVEL4_CONFIDENT_THRESH:
        family_factor = config.LEVEL4_ACTION_ACTUATION_DOWNWEIGHT

    w_adjusted = w * family_factor
    final_action = (1 - w_adjusted) * policy_action + w_adjusted * recovery_action
    return final_action.astype(np.float32), class_probs, w_adjusted
