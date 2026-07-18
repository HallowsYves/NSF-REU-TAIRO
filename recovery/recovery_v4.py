"""
Recovery v4 Tier 1: Classifier-Conditioned Adaptive Recovery (CCAR).

Design source: RECOVERY_V4.md section 2 (Tier 1). Additive to v1-v3, not a
replacement -- see CLAUDE.md section 10 and 12. Tier 2 (trained recovery
experts) is explicitly out of scope; do not add anything here in that
direction.

This module implements section 2.1 (trigger), the shared class-probability
helper, section 2.2's five recovery experts and mixture function, and
section 2.3's per-step recovery_step. episode_runner.py integration is
wired in behind the "sac_her_recovery_v4" method name, additive only.

Label-name bug found and fixed while wiring the mixture (2026-07-15):
RECOVERY_V4.md 2.2's expert dict uses the key 'object_pose_spoof' for
relocalization_expert. 'object_pose_spoof' is an ATTACK CONDITION name
(config.ALL_CONDITIONS), not a failure-mode LABEL -- the classifier's
class_probs dict is keyed by evaluation.failure_mode_labeling.ALL_LABELS,
which has 'spoofed_goal', not 'object_pose_spoof'. Using the literal doc
key would make class_probs.get('object_pose_spoof', 0.0) always return
0.0, silently disabling relocalization_expert entirely -- the same class
of bug as the classes_/label_order ordering mismatch get_class_probs
exists to prevent. Fixed to 'spoofed_goal' below. Flagged in the Phase 3
report rather than silently carried over from the design doc.

Tier 1 CCAR is currently scoped to the clean_2M checkpoint only -- see the
2026-07-15 addendum in RECOVERY_V4.md section 2.6 and the Phase 1 findings
in findings.md for why the other three checkpoints are out of scope.

Signature note (expert functions): RECOVERY_V4.md 2.2 writes each expert as
expert_fn(trusted_state, feature_vec). That is schematic. relocalization_expert
(median-filtered object position over the last k steps) and regrasp_expert
(retreat-and-reapproach anchored on the last known good state) both need
short per-episode history that neither trusted_state (single-step) nor
feature_vec (scalar aggregates only) carries. All five experts here take a
third argument, expert_state (an ExpertState instance), for consistency of
the dispatch loop even though three of the five do not read it. This is a
deliberate elaboration of the design doc, not a silent deviation -- flagged
in the Phase 2 report.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from config import GRASP_DIST_THRESHOLD, GRASP_WINDOW, CAUSAL_WINDOW_SHORT

# Phase 5b live-data calibration refit (2026-07-15). alpha and the steepness
# multiplier k were originally set from a synthetic p_success sequence
# (Phase 1). Refit from 40 live clean_2M clean episodes (Phase 5a) because
# per-step classifier queries at steps outside its 14-checkpoint training
# distribution add noise to ema_pfail that the synthetic sequence did not
# have. See findings.md "Phase 5b" for the before/after numbers and the
# full reasoning, including why k went UP not down (clean ema sits below
# the midpoint, where a steeper sigmoid lowers w). Only alpha and k changed;
# the midpoint (per-checkpoint clean_pfail_p95) is untouched from Phase 1.
DEFAULT_ALPHA = 1.0 / CAUSAL_WINDOW_SHORT   # 0.05 -- EMA window (~20 steps)
                                            # matches the classifier's short
                                            # causal window, vs the ~10-step
                                            # window of the old alpha=0.1 that
                                            # under-smoothed 150-step episodes.


def get_class_probs(model, x: np.ndarray) -> Dict[str, float]:
    """Return {class_label: probability} for one row, using the model's own
    class ordering rather than any externally saved label list.

    RandomForestClassifier.classes_ is sorted alphabetically by sklearn and
    does NOT match the semantic order some callers might expect (e.g. a
    saved 'label_order' list with 'success' listed first). Zipping
    predict_proba's output against the wrong order silently mislabels every
    probability. This helper is the single place that ordering is handled,
    so every caller (trigger calibration, the Tier 1 experts, the step
    function) reads probabilities the same, correct way.

    Args:
        model: fitted sklearn classifier exposing classes_ and predict_proba.
        x:     single row, shape (1, n_features).

    Returns:
        Dict mapping each class label string to its predicted probability.
    """
    probs = model.predict_proba(x)[0]
    return dict(zip(model.classes_, probs))


class TriggerWeight:
    """EMA-smoothed, data-calibrated blend weight (RECOVERY_V4.md 2.1).

    Replaces v1-v3's tau_sat / N_sustain / P_exit hard-threshold-and-counter
    detector with a continuous weight in [0, 1] derived from the classifier's
    pooled success/fail probability.

    One instance per (episode, model checkpoint) -- clean_pfail_p95 is
    calibrated per trained model checkpoint (clean_2M / clean_500k /
    randomized_2M / randomized_500k), not pooled across checkpoints, since
    baseline clean-condition failure behavior differs enormously between
    checkpoints (see calibration script and CLAUDE.md section 5).
    """

    # Steepness multiplier k. Two-stage history:
    #   1. The literal RECOVERY_V4.md default (steepness = 1 / (midpoint + eps))
    #      gives steepness * midpoint ~= 1 for any midpoint, so
    #      w(ema_pfail=0) = 1/(1+e^1) ~= 0.269 -- a ~27% blend on a perfectly
    #      clean episode, contradicting the design's w-near-zero goal. Phase 1
    #      fixed this to steepness = K_STEEPNESS / (midpoint + eps), K=4.
    #   2. Phase 5b (2026-07-15) refit K from 4 to 6 using the live clean_2M
    #      ema_pfail distribution (Phase 5a), targeting w ~= 0.05 at the
    #      observed clean ema_pfail p95 (0.5103). Note the direction: clean
    #      ema_pfail sits well BELOW the midpoint (0.985), i.e. on the
    #      sigmoid's lower tail, where a STEEPER sigmoid (larger K) drives w
    #      DOWN. Lowering K would have raised clean w, not lowered it. The
    #      attacked-condition ema_pfail (~1.0, near the midpoint) is barely
    #      affected by K, so the separation is preserved. See findings.md
    #      "Phase 5b" for the before/after numbers.
    K_STEEPNESS = 6

    def __init__(self, clean_pfail_p95, alpha=DEFAULT_ALPHA, steepness=None):
        self.alpha = alpha
        self.ema_pfail = 0.0
        self.midpoint = clean_pfail_p95
        self.steepness = steepness or (self.K_STEEPNESS / (clean_pfail_p95 + 1e-6))

    def update(self, p_success):
        p_fail = 1 - p_success
        self.ema_pfail = self.alpha * p_fail + (1 - self.alpha) * self.ema_pfail
        w = 1 / (1 + math.exp(-self.steepness * (self.ema_pfail - self.midpoint)))
        return w


def build_trusted_state(obs: Dict) -> Dict[str, np.ndarray]:
    """Decode the raw, unattacked obs dict into the trusted_state shape the
    Tier 1 experts expect.

    Uses the exact same slicing as evaluation/episode_runner.py's
    _pnp_spatial_fields (gripper_pos=obs_vec[0:3], object_pos=obs_vec[3:6],
    object_velp=obs_vec[14:17], grip_velp=obs_vec[20:23]) -- the canonical
    decode already used by the recovery v2/v3 call site (raw obs, never
    policy_obs; CLAUDE.md section 7/9). Not re-derived independently, so
    the two paths cannot silently drift apart.

    PickAndPlace only (25-dim observation). Tier 1 CCAR does not apply to
    FetchReach.
    """
    obs_vec = np.asarray(obs["observation"], dtype=np.float64)
    return {
        "gripper_pos": obs_vec[0:3].copy(),
        "object_pos": obs_vec[3:6].copy(),
        "object_velp": obs_vec[14:17].copy(),
        "grip_velp": obs_vec[20:23].copy(),
        "gripper_aperture": float(np.sum(obs_vec[9:11])),
        "true_goal": np.asarray(obs["desired_goal"], dtype=np.float64).copy(),
    }


@dataclass
class ExpertState:
    """Per-episode rolling state for the Tier 1 experts.

    Instantiate once per episode (mirrors v2/v3's RecoveryState). Call
    update() once per step, before invoking any expert, so the buffers
    only ever reflect steps <= current -- no forward-looking information,
    same causal discipline as evaluation/causal_features.py.
    """
    median_window: int = 5
    object_pos_history: deque = field(default_factory=lambda: deque(maxlen=5))
    last_good_gripper_pos: Optional[np.ndarray] = None
    last_good_object_pos: Optional[np.ndarray] = None
    last_good_step: int = -1

    def __post_init__(self):
        if self.object_pos_history.maxlen != self.median_window:
            self.object_pos_history = deque(maxlen=self.median_window)

    def reset(self):
        self.object_pos_history.clear()
        self.last_good_gripper_pos = None
        self.last_good_object_pos = None
        self.last_good_step = -1

    def update(self, trusted_state: Dict, feature_vec: Dict, step: int) -> None:
        """Refresh rolling history. Call once per step, before the experts run.

        "Good state" = a real grasp is (or was) actively in progress this
        episode: the kinematic contact streak has reached GRASP_WINDOW and
        the object has not since separated past the drop threshold. Both
        are causal, scalar feature_vec fields already computed by
        evaluation/causal_features.py -- no hindsight involved, no new
        detection logic invented here.
        """
        self.object_pos_history.append(
            np.asarray(trusted_state["object_pos"], dtype=np.float64).copy()
        )

        grasp_in_progress = (
            feature_vec.get("contact_streak_now", 0) >= GRASP_WINDOW
            and not feature_vec.get("separated_after_contact_now", 0.0)
        )
        if grasp_in_progress:
            self.last_good_gripper_pos = np.asarray(
                trusted_state["gripper_pos"], dtype=np.float64
            ).copy()
            self.last_good_object_pos = np.asarray(
                trusted_state["object_pos"], dtype=np.float64
            ).copy()
            self.last_good_step = step

    def median_object_pos(self, fallback) -> np.ndarray:
        """Per-axis median of the object position history, or fallback if empty."""
        if len(self.object_pos_history) == 0:
            return np.asarray(fallback, dtype=np.float64)
        return np.median(np.stack(list(self.object_pos_history)), axis=0)


# ---------------------------------------------------------------------------
# Tier 1 experts (RECOVERY_V4.md 2.2) -- deterministic, feature-based control
# rules. Not trained models; no recovery-action ground truth is required.
# PickAndPlace-only (4-D action space: [dx, dy, dz, gripper_ctrl]).
# ---------------------------------------------------------------------------

ACTION_DIM = 4

# Gripper ctrl sign confirmed against gymnasium_robotics fetch_env.py
# _set_action(): action[3] is broadcast to both fingers identically and
# passed straight to ctrl_set_action -- standard MujocoFetch convention,
# positive opens, negative closes.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
GRIPPER_HOLD = 0.0  # no additional open/close command; let current width settle

# Proportional gain matches policies/rule_based_policy.py's DEFAULT_GAIN=5.0,
# for consistency with the existing v1-v3 rule-based controller ("same
# spirit ... just split into specialized versions", RECOVERY_V4.md 2.2).
# Derivative gains and the stabilize/regrasp constants are Tier 1
# hyperparameters -- acceptable as constants per RECOVERY_V4.md 2.4.
KP_APPROACH = 5.0
KD_APPROACH = 1.0
KP_TRANSPORT = 5.0
KD_TRANSPORT = 1.0
KP_STABILIZE = 2.0
KD_STABILIZE = 3.0
KP_REGRASP = 5.0
KP_RELOC = 5.0
KD_RELOC = 1.0


def approach_expert(trusted_state: Dict, feature_vec: Dict, expert_state: ExpertState) -> np.ndarray:
    """PD control toward the object position (never_reached_object).

    Reads trusted_state: gripper_pos, object_pos, object_velp, grip_velp.
    Reads feature_vec: none -- pure state-based PD.
    Does not read expert_state.

    Gripper commanded open: the object has not been reached yet, so there
    is nothing to hold onto.
    """
    gripper_pos = np.asarray(trusted_state["gripper_pos"], dtype=np.float64)
    object_pos = np.asarray(trusted_state["object_pos"], dtype=np.float64)
    object_velp = np.asarray(trusted_state["object_velp"], dtype=np.float64)
    grip_velp = np.asarray(trusted_state["grip_velp"], dtype=np.float64)

    error = object_pos - gripper_pos
    d_error = object_velp - grip_velp  # d(error)/dt

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:3] = KP_APPROACH * error + KD_APPROACH * d_error
    action[3] = GRIPPER_OPEN
    return action


def transport_expert(trusted_state: Dict, feature_vec: Dict, expert_state: ExpertState) -> np.ndarray:
    """PD control toward the trusted true goal (divergent_transport).

    Reads trusted_state: object_pos, object_velp, true_goal.
    Reads feature_vec: anrm_std_short (recent action-norm variability),
    used to scale the gain down when the base policy is already jittery --
    RECOVERY_V4.md 2.2's "gain scaled by the inverse of recent action-norm
    variance."
    Does not read expert_state.

    The object is assumed already grasped (divergent_transport, by the
    taxonomy's ordering, requires a prior confirmed grasp): steering the
    object toward the true goal is equivalent to steering the gripper the
    same way, since the arm carries the held object. Gripper commanded
    closed -- maintain the grasp during transport.
    """
    object_pos = np.asarray(trusted_state["object_pos"], dtype=np.float64)
    object_velp = np.asarray(trusted_state["object_velp"], dtype=np.float64)
    true_goal = np.asarray(trusted_state["true_goal"], dtype=np.float64)

    error = true_goal - object_pos
    d_error = -object_velp  # d(error)/dt when true_goal is fixed

    anrm_std = feature_vec.get("anrm_std_short", 0.0)
    gain_scale = 1.0 / (1.0 + anrm_std)

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:3] = gain_scale * (KP_TRANSPORT * error + KD_TRANSPORT * d_error)
    action[3] = GRIPPER_CLOSE
    return action


def relocalization_expert(trusted_state: Dict, feature_vec: Dict, expert_state: ExpertState) -> np.ndarray:
    """PD control toward a median-filtered object position estimate (object_pose_spoof).

    Reads trusted_state: gripper_pos, grip_velp, object_velp, object_pos
    (fallback only, used solely when expert_state has no history yet).
    Reads feature_vec: contact_streak_now (choose open vs. closed gripper).
    Reads expert_state: object_pos_history -- the last
    expert_state.median_window raw trusted_state["object_pos"] readings,
    maintained by expert_state.update() every step. This is the
    "median-filtered estimate over the last k steps of trusted_state"
    RECOVERY_V4.md 2.2 calls for.

    Per-axis median (not mean) is deliberate: a single-step spoof offset is
    an outlier the median rejects, whereas a mean would still be pulled
    toward it.
    """
    gripper_pos = np.asarray(trusted_state["gripper_pos"], dtype=np.float64)
    grip_velp = np.asarray(trusted_state["grip_velp"], dtype=np.float64)
    object_velp = np.asarray(trusted_state["object_velp"], dtype=np.float64)

    robust_object_pos = expert_state.median_object_pos(fallback=trusted_state["object_pos"])

    error = robust_object_pos - gripper_pos
    d_error = object_velp - grip_velp

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:3] = KP_RELOC * error + KD_RELOC * d_error
    action[3] = GRIPPER_CLOSE if feature_vec.get("contact_streak_now", 0) >= GRASP_WINDOW else GRIPPER_OPEN
    return action


def grasp_stabilize_expert(trusted_state: Dict, feature_vec: Dict, expert_state: ExpertState) -> np.ndarray:
    """Damp gripper velocity and hold current finger width (reached_but_failed_grasp).

    Reads trusted_state: gripper_pos, object_pos, grip_velp.
    Reads feature_vec: dto_now (gripper-to-object distance now) -- adds a
    small closing nudge only while still outside GRASP_DIST_THRESHOLD;
    once within contact range the expert holds rather than pushes further.
    Does not read expert_state.

    RECOVERY_V4.md 2.2: "reduce gripper velocity and hold current
    finger-width." Gripper commanded to hold (0.0) rather than forced
    fully closed -- transport_expert (or the base policy) closes the grip
    once a stable contact streak is established.
    """
    gripper_pos = np.asarray(trusted_state["gripper_pos"], dtype=np.float64)
    object_pos = np.asarray(trusted_state["object_pos"], dtype=np.float64)
    grip_velp = np.asarray(trusted_state["grip_velp"], dtype=np.float64)

    dto_now = feature_vec.get("dto_now", 0.0)
    nudge = np.zeros(3, dtype=np.float64)
    if dto_now > GRASP_DIST_THRESHOLD:
        nudge = KP_STABILIZE * (object_pos - gripper_pos)

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:3] = nudge - KD_STABILIZE * grip_velp
    action[3] = GRIPPER_HOLD
    return action


def regrasp_expert(trusted_state: Dict, feature_vec: Dict, expert_state: ExpertState) -> np.ndarray:
    """Retreat-and-reapproach anchored on the last known good state (grasped_but_dropped).

    Reads trusted_state: gripper_pos, plus whatever approach_expert reads
    in the bootstrap fallback case.
    Reads feature_vec: none directly here -- the "good" determination
    happens in expert_state.update(), not in this function.
    Reads expert_state: last_good_gripper_pos -- the gripper position
    recorded the last time a real grasp was in progress this episode (a
    rolling "last known good state," not a hardcoded distance, per
    RECOVERY_V4.md 2.2).

    If no good state has been recorded yet this episode (bootstrap case --
    grasped_but_dropped firing before any confirmed grasp, which the
    taxonomy's ordering should prevent but is handled defensively here),
    falls back to approach_expert's behavior toward the current object
    position.
    """
    gripper_pos = np.asarray(trusted_state["gripper_pos"], dtype=np.float64)

    if expert_state.last_good_gripper_pos is None:
        return approach_expert(trusted_state, feature_vec, expert_state)

    error = expert_state.last_good_gripper_pos - gripper_pos

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:3] = KP_REGRASP * error
    action[3] = GRIPPER_OPEN
    return action


# ---------------------------------------------------------------------------
# Mixture (RECOVERY_V4.md 2.2) and step function (2.3)
# ---------------------------------------------------------------------------

# Keyed by evaluation.failure_mode_labeling.ALL_LABELS entries -- see the
# module docstring for the 'object_pose_spoof' -> 'spoofed_goal' fix.
EXPERTS = {
    "divergent_transport":      transport_expert,
    "spoofed_goal":             relocalization_expert,
    "reached_but_failed_grasp": grasp_stabilize_expert,
    "grasped_but_dropped":      regrasp_expert,
    "never_reached_object":     approach_expert,
}

# Below this blend weight, skip expert computation entirely and pass the
# base policy action through unchanged. Not specified numerically in
# RECOVERY_V4.md 2.3 ("skip expert computation when clearly clean") --
# set to match the Phase 1 clean-episode acceptance target (w < 0.05).
EPSILON = 0.05


def compute_recovery_action(trusted_state: Dict, class_probs: Dict[str, float],
                             feature_vec: Dict, expert_state: ExpertState) -> np.ndarray:
    """Weighted sum of expert actions, weighted by the classifier's per-class
    probability (RECOVERY_V4.md 2.2). Not a model selector -- every expert
    with nonzero probability contributes proportionally.

    'success' has no corresponding expert (there is nothing to recover
    from) and is skipped via .get(cls, 0.0) below along with any other
    label not in EXPERTS.
    """
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    for cls, expert_fn in EXPERTS.items():
        weight = class_probs.get(cls, 0.0)
        if weight <= 0.0:
            continue
        action += weight * expert_fn(trusted_state, feature_vec, expert_state)
    return action


def recovery_step(policy_action: np.ndarray, obs: Dict, step_history_df,
                   classifier_artifact: Dict, trigger: TriggerWeight,
                   expert_state: ExpertState, step: int):
    """One step of Tier 1 CCAR (RECOVERY_V4.md 2.3).

    Args:
        policy_action:       the base SAC+HER action for this step (already
                              past any action-level attack -- same
                              "executed_action" v2/v3 receive).
        obs:                  raw, unattacked observation dict for this step
                              (never policy_obs -- CLAUDE.md section 7/9).
        step_history_df:      DataFrame of step-log rows for steps
                              [0, step-1] (this step has not happened yet).
                              Empty / step==0 is the step-0 guard, mirroring
                              v2/v3.
        classifier_artifact:  the loaded online_failure_classifier.pkl dict
                              ({'model', 'feature_cols', ...}).
        trigger:              TriggerWeight instance for this episode's
                              model checkpoint (see calibration script).
        expert_state:         ExpertState instance for this episode.
        step:                 current timestep index (0-indexed).

    Returns:
        (final_action, class_probs, w) -- final_action is float32 ndarray,
        class_probs is {} at the step-0 guard, w is 0.0 at the step-0 guard.

    Elaboration on RECOVERY_V4.md 2.3, flagged: expert_state.update() runs
    on every non-guarded step regardless of w, not only when w >= EPSILON.
    The design doc's pseudocode computes trusted_state/expert bookkeeping
    only in the non-skipped branch, but ExpertState's rolling "last known
    good state" must be captured exactly when things ARE going well (w is
    small) -- skipping the update whenever w < EPSILON would mean
    last_good_gripper_pos could never be set during an actual good grasp,
    defeating the buffer's purpose. Updating it is cheap (array slicing,
    no classifier call), so this doesn't cost the "skip expensive work
    when clean" optimization the doc's early-return is actually after.
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

    recovery_action = compute_recovery_action(trusted_state, class_probs, feature_vec, expert_state)
    final_action = (1 - w) * policy_action + w * recovery_action
    return final_action.astype(np.float32), class_probs, w
