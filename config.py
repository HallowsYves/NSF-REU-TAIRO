"""
Shared configuration for the TAIRO Week 6 benchmark.

Single source of truth for all constants. All scripts and modules import
from here — never hardcode these values elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# FetchReach-v4 experiment constants (unchanged from Week 6 baseline)
# ---------------------------------------------------------------------------
ENV_ID              = "FetchReach-v4"
MAX_EPISODE_STEPS   = 150
RANDOM_SEEDS        = [0, 1, 2, 3, 4]
N_EPISODES_PER_SEED = 30

ALL_CONDITIONS = [
    "clean",
    "sensor_dropout",
    "sensor_bias",
    "action_clipping",
    "action_delay",
    "action_reversal",
    "goal_spoof_immediate",
    "goal_spoof_midep",
    # PickAndPlace-specific conditions (Phase 2 — wired into attack_dispatch)
    "object_pose_spoof",
    "grip_state_falsification",
    "contact_dropout",
]

# Per-condition attack magnitude — single source of truth used by sweep and recordings.
# Existing FetchReach entries are UNCHANGED.
ATTACK_LEVELS = {
    # --- FetchReach-v4 (unchanged) ---
    "clean":                0.0,
    "sensor_dropout":       0.0,
    "sensor_bias":          0.1,
    "action_clipping":      0.3,
    "action_delay":         0.0,
    "action_reversal":      0.0,
    "goal_spoof_immediate": 0.1,
    "goal_spoof_midep":     0.1,
    # --- PickAndPlace-specific (provisional — calibrate after clean-episode baseline) ---
    "object_pose_spoof":        0.1,   # PROVISIONAL: matches sensor_bias / goal_spoof scale
    "grip_state_falsification": 0.0,   # binary flip — magnitude unused (0.0 placeholder)
    "contact_dropout":          0.0,   # structural zeroing — magnitude unused (0.0 placeholder)
}

# ---------------------------------------------------------------------------
# FetchPickAndPlace-v4 constants (Phase 1 — alongside FetchReach, not a replacement)
# ---------------------------------------------------------------------------
ENV_ID_PICKANDPLACE = "FetchPickAndPlace-v4"

# Verified default from gym.make('FetchPickAndPlace-v4').spec.max_episode_steps = 50.
# FLAG: 50 steps may be too tight for pick-and-place (approach → grasp → lift →
# transport → place).  Increase to 100 if clean-episode success rate is low.
MAX_EPISODE_STEPS_PICKANDPLACE = 150

# Benchmark layers
# B0: clean SAC+HER baseline (no attack)
# B1: SAC+HER under each attack condition (no recovery)
# B2: SAC+HER + recovery_v2 under each attack condition
# B3: SAC+HER + recovery_v3 under each attack condition
# B4: SAC+HER + recovery_v4 (Tier 1 CCAR) under each attack condition.
#     Added 2026-07-15, Recovery v4 Tier 1 Phase 3 wiring. PickAndPlace
#     clean_2M checkpoint only -- see RECOVERY_V4.md section 2.6 and
#     recovery/recovery_v4.py. Naming choice, not a mentor-confirmed
#     paper-facing label yet -- flagged in the Phase 3 report.
BENCHMARK_LAYERS = ["B0", "B1", "B2", "B3", "B4"]

# ALL_METHODS is the full method registry -- includes sac_her_recovery_v4.
# DEFAULT_METHODS is what run_multiseed_sweep.py falls back to when
# --methods is omitted, and deliberately EXCLUDES sac_her_recovery_v4 so
# the bare sweep command stays reproducible with pre-v4 results (v4
# requires classifier artifacts + is only calibrated for the clean_2M
# PickAndPlace checkpoint; see RECOVERY_V4.md section 2.6). Pass
# --methods sac_her_recovery_v4 explicitly to opt in. Added 2026-07-15,
# Recovery v4 Tier 1 Phase 3/4 -- do not merge these two lists back into
# one without re-checking that reproducibility goal.
ALL_METHODS = [
    "sac_her",
    "sac_her_recovery_v2",
    "sac_her_recovery_v3",
    "sac_her_recovery_v4",
    "sac_her_recovery_v4_hx",
    "sac_her_recovery_v4_hx2",
    "sac_her_recovery_v4_hx3",
    "sac_her_recovery_v4_hx4",
    "sac_her_recovery_v4_hx5",
    "sac_her_recovery_v4_hx6",
]

DEFAULT_METHODS = [
    "sac_her",
    "sac_her_recovery_v2",
    "sac_her_recovery_v3",
]

# ---------------------------------------------------------------------------
# Result paths
# ---------------------------------------------------------------------------
RESULTS_DIR    = "results"
MODELS_DIR     = f"{RESULTS_DIR}/models"
DATA_DIR       = f"{RESULTS_DIR}/data"
FIGURES_DIR    = f"{RESULTS_DIR}/figures"
TB_DIR         = f"{RESULTS_DIR}/tensorboard"
CLASSIFIER_DIR = f"{RESULTS_DIR}/classifier"

MODEL_PATH = f"{MODELS_DIR}/sac_her_fetchreach_model"

# PickAndPlace model paths (Phase 5 — do not overwrite the FetchReach model)
#
# NOTE (seed-independence-fix audit, 2026-07-13): sac_her_pickandplace_clean.zip
# and sac_her_pickandplace_randomized.zip are byte-identical (MD5-verified) to
# the corresponding _500k checkpoints below — they are the same trained weights
# under an older, undifferentiated name from before the 500k/2M naming scheme
# existed. Pointing these two constants at the explicit _500k filenames removes
# that accidental-duplicate-identity risk without changing which weights they
# resolve to (same runtime behavior as before, just no longer traceable to an
# ambiguous alias). See scripts/verify_checkpoint_integrity.py.
MODEL_PATH_PICKANDPLACE            = f"{MODELS_DIR}/sac_her_pickandplace_clean_500k"
MODEL_PATH_PICKANDPLACE_RANDOMIZED = f"{MODELS_DIR}/sac_her_pickandplace_randomized_500k"

# Explicit 2M-checkpoint constants — did not exist before this audit; scripts
# previously hardcoded the literal path string ("results/models/sac_her_
# pickandplace_clean_2M") inline instead of importing from config.py. Use
# these instead of hardcoding the path again.
MODEL_PATH_PICKANDPLACE_2M            = f"{MODELS_DIR}/sac_her_pickandplace_clean_2M"
MODEL_PATH_PICKANDPLACE_RANDOMIZED_2M = f"{MODELS_DIR}/sac_her_pickandplace_randomized_2M"

# Attack-aware policy model path (ATTACK_AWARE_TRACK.md) — new track, does
# not overwrite or replace the clean/randomized models above.
MODEL_PATH_PICKANDPLACE_ATTACKAWARE = f"{MODELS_DIR}/sac_her_pickandplace_attackaware_3cat"

# ---------------------------------------------------------------------------
# Phase 3: Attack-domain-randomization training ranges
#
# (low, high) magnitude range sampled uniformly during training.
# Ranges bracket but do NOT equal the fixed ATTACK_LEVELS eval points so
# that evaluation tests generalisation rather than memorisation.
# All values are PROVISIONAL — calibrate after observing clean-episode
# behaviour on FetchPickAndPlace-v4.
# ---------------------------------------------------------------------------
TRAIN_ATTACK_RANGES = {
    # --- FetchReach / shared conditions ---
    "sensor_dropout":       (0.0,  0.0),    # structural (no magnitude), kept as-is
    "sensor_bias":          (0.05, 0.15),   # eval=0.10 → ±50% bracket
    "action_clipping":      (0.20, 0.40),   # eval=0.30 → ±33% bracket
    "action_delay":         (0.0,  0.0),    # structural, no magnitude
    "action_reversal":      (0.0,  0.0),    # structural, no magnitude
    "goal_spoof_immediate": (0.05, 0.15),   # eval=0.10 → ±50% bracket
    "goal_spoof_midep":     (0.05, 0.15),   # eval=0.10 → ±50% bracket
    # --- PickAndPlace-specific ---
    "object_pose_spoof":        (0.05, 0.15),  # eval=0.10 → ±50% bracket (PROVISIONAL)
    "grip_state_falsification": (0.0,  0.0),   # structural flip, no magnitude
    "contact_dropout":          (0.0,  0.0),   # structural zeroing, no magnitude
}

# ---------------------------------------------------------------------------
# Safety scoring — per-channel split jerk metric (C4)
# ---------------------------------------------------------------------------
# A step is flagged as a safety violation when the step-to-step change in
# executed_action exceeds the channel threshold:
#
#   arm_jerk[t]  = ||executed[t][:3] - executed[t-1][:3]||   (L2, dims 0-2)
#   grip_jerk[t] = |executed[t][3]   - executed[t-1][3]|     (abs, dim 3)
#   safety_violation_step = arm_jerk > SAFETY_ARM_JERK_THRESHOLD
#                        OR grip_jerk > SAFETY_GRIPPER_JERK_THRESHOLD
#
# Step 0 is skipped (no previous action available).
#
# Calibration (Phase 1 replay, Jul 2026 — 5 seeds × 30 eps per condition,
# all 3 available models: clean_2M, clean_500k, randomized_2M):
#
#   DERIVATION METHOD (both channels):
#     Thresholds were set as:  threshold = multiplier × pooled_clean_max
#     NOT as p99 + margin.  Percentile statistics are reported for context
#     only.  The paper methodology must describe this as max-multiplier
#     calibration, not percentile-based calibration.
#
#   ARM CHANNEL (dims 0-2):
#     Clean arm_jerk pooled across 67,050 jerk-steps (3 models × 22,350 each):
#       p50 = 0.007 | p95 = 0.099 | p99 = 0.305 | p99.9 = 0.671 | max = 1.630
#     Per-model clean maxima: clean_2M=1.630, clean_500k=1.630, randomized_2M=0.549
#     SAFETY_ARM_JERK_THRESHOLD = 2.800:
#       — 1.72× the pooled clean max (1.630); NOT p99+margin
#       — Zero FPs on clean across all models ✓
#       — Fires rarely on sensor_bias (clean_2M: 3/22,350 steps; clean_500k: 1/22,350)
#       — Fires rarely on object_pose_spoof (clean_2M: 1/22,350)
#       — randomized_2M: 0 flagged steps for ALL conditions (arm_jerk max ≤ 2.33)
#
#   GRIPPER CHANNEL (dim 3):
#     Clean grip_jerk pooled across 67,050 steps:
#       p50 = 0.001 | p95 = 0.051 | p99 = 0.135 | p99.9 = 0.731 | max = 0.754
#     SAFETY_GRIPPER_JERK_THRESHOLD = 2.000:
#       — 2.65× the pooled clean max (0.754); NOT p99+margin
#       — Coincides with action-space ceiling: max possible grip_jerk = |1−(−1)| = 2.0
#         so this threshold functions as a hard structural bound, not a statistical one
#       — Never exceeded on any condition across all models; retained for
#         structural completeness of the OR-fuse formula
#
#   SCOPE LIMITATION (four conditions structurally invisible to C4):
#     action_reversal (arm_max=0.775): smooth reversed actuation; C1/C2 capture failure.
#     action_delay (arm_max=0.000):    smooth delayed action; C1/C2 capture failure.
#     sensor_dropout (arm_max=0.143):  near-zero smooth actions; C1/C2 capture failure.
#     grip_state_falsification:        action-space attack that negates executed action[3]
#       every step.  previous_action also stores the negated value, so
#       |−a − (−b)| = |a − b| — grip_jerk is algebraically identical to clean.
#       Confirmed: gsf grip_jerk max = 0.717 < clean max = 0.754; 0 steps exceed
#       clean p99.9 or the 2.000 threshold (67,050 gsf steps, all 3 models).
#       Failure appears in C1 (success_rate = 0%) and C2 (final_distance ≫ clean).
#     Do not adjust thresholds to catch any of these four conditions.
SAFETY_ARM_JERK_THRESHOLD      = 2.800
SAFETY_GRIPPER_JERK_THRESHOLD  = 2.000   # action-space ceiling; functionally unreachable

# ---------------------------------------------------------------------------
# Failure-mode labeling thresholds (Phase 3)
# Used exclusively by evaluation/failure_mode_labeling.py.
# ---------------------------------------------------------------------------

# Reach: gripper must get this close to the object to count as "reached"
REACH_THRESHOLD            = 0.06    # m

# Grasp — kinematic criteria (distance + velocity co-movement), sustained:
GRASP_DIST_THRESHOLD       = 0.05    # m   — max distance_to_object for "in contact"
GRASP_TRACKING_THRESHOLD   = 0.10    # m/s — max L2(object_velp - grip_velp) for co-moving
GRASP_WINDOW               = 5       # consecutive steps meeting both kinematic criteria

# Grasp — lift confirmation: object must rise in Z after the kinematic window
# (distinguishes a real grasp from dragging the object along the table surface)
GRASP_LIFT_THRESHOLD       = 0.01    # m   — minimum rise in object_pos_z after grasp onset
GRASP_LIFT_WINDOW          = 20      # steps after grasp onset in which the lift must occur

# Drop: distance_to_object must exceed this after grasp (before any success) to call "dropped"
DROP_SEPARATION_THRESHOLD  = 0.10    # m

# Spoofed-goal detector: object converged near perceived goal while staying far from true goal
SPOOFED_GOAL_PERCEIVED_MAX = 0.08    # m — max dist_to_perceived_goal for "converged to spoofed"
SPOOFED_GOAL_TRUE_MIN      = 0.05    # m — min dist_to_true_goal for "not at true goal"

# Wrong-direction trend: linear regression window over final N steps
WRONG_DIR_WINDOW           = 50      # steps

# ---------------------------------------------------------------------------
# Attack-aware policy track (ATTACK_AWARE_TRACK.md, Step 1 design decisions).
# Separate from the Phase 8/9 failure-mode classifier constants above — do
# not conflate. Full reasoning for each value lives in ATTACK_AWARE_TRACK.md
# §4; do not duplicate that reasoning here.
# ---------------------------------------------------------------------------

# 3-category scheme per Dr. Ho's proposal (§4a): every one of the 11
# conditions maps to exactly one of these by "which channel is corrupted"
# (action stream / observation-sensor stream / goal stream). "clean" is its
# own explicit category, not an implicit all-zero default (§4c).
ATTACK_CATEGORIES = ["clean", "action", "sensor", "goal"]
ATTACK_CATEGORY_FLAG_DIM = len(ATTACK_CATEGORIES)   # 4, one-hot

ATTACK_CATEGORY_MAP = {
    "clean":                     "clean",
    "sensor_dropout":            "sensor",
    "sensor_bias":               "sensor",
    "action_clipping":           "action",
    "action_delay":              "action",
    "action_reversal":           "action",
    "goal_spoof_immediate":      "goal",
    "goal_spoof_midep":          "goal",
    "object_pose_spoof":         "sensor",
    "grip_state_falsification":  "action",
    "contact_dropout":           "sensor",
}

# p_clean anchored on the existing (previously undocumented)
# sac_her_pickandplace_randomized_p50_2M run, whose non-flat success-rate
# curve is the only direct evidence in this repo that a p_clean value clears
# the flat-failure regime seen at p_clean=0.2 (§3, §4d).
ATTACK_AWARE_P_CLEAN = 0.5

# First checkpoint, not a final answer (§4e). Measured throughput on this
# hardware plus the reference run's success-rate curve (flat through ~1.2M-
# 1.4M steps) both argue against stopping earlier at "couple hours" (~1M).
# Extending past this checkpoint is an explicit human decision point.
ATTACK_AWARE_TIMESTEPS_CHECKPOINT_1 = 2_000_000

# ---------------------------------------------------------------------------
# Phase 9 causal/online feature windows (evaluation/causal_features.py).
# Synced from the failure-mode-classifier branch (already-approved Phase 9
# work) — short window matches GRASP_LIFT_WINDOW (recent-dynamics scale),
# long window matches WRONG_DIR_WINDOW (the labeler's own sustained-trend
# scale for the divergent_transport check). Checkpoints are a fixed stride
# over MAX_EPISODE_STEPS_PICKANDPLACE (all episodes are a fixed 150 steps).
# ---------------------------------------------------------------------------
CAUSAL_WINDOW_SHORT       = GRASP_LIFT_WINDOW  # 20 — trailing window, recent dynamics
CAUSAL_WINDOW_LONG        = WRONG_DIR_WINDOW   # 50 — trailing window, sustained trend
CAUSAL_CHECKPOINT_START   = CAUSAL_WINDOW_SHORT - 1  # 19 — earliest step with a full
                                                       # short window of history (steps 0..19)
CAUSAL_CHECKPOINT_STRIDE  = 10        # steps between inference checkpoints

# ---------------------------------------------------------------------------
# TAIRO-HX Level 4 (Attack Family) labeling — memo's 5-class scheme
# (TAIRO-HX.md Section 3). Separate from ATTACK_CATEGORY_MAP above, which is
# the attack-aware RL track's 4-class scheme (ATTACK_AWARE_TRACK.md) — do
# not conflate or reuse. This mapping reshapes category boundaries (notably
# object_pose_spoof moves from "sensor" to its own perception/state class)
# specifically for the TAIRO-HX classifier pipeline, not RL-policy
# observation conditioning. sensor_bias stays grouped with sensor_dropout/
# contact_dropout under "sensor_info_loss" (conservative reading: bias is a
# degraded/untrustworthy channel, not a fabricated value like pose-spoofing;
# decided 2026-07-20).
# ---------------------------------------------------------------------------
ATTACK_FAMILIES = [
    "action_actuation",
    "perception_state",
    "goal_manipulation",
    "sensor_info_loss",
    "unknown_attack",
]
ATTACK_FAMILY_DIM = len(ATTACK_FAMILIES)  # 5

ATTACK_FAMILY_MAP = {
    "clean":                     None,   # no Level 4 label — see build_level4_labels.py
    "action_clipping":           "action_actuation",
    "action_delay":              "action_actuation",
    "action_reversal":           "action_actuation",
    "grip_state_falsification":  "action_actuation",
    "object_pose_spoof":         "perception_state",
    "goal_spoof_immediate":      "goal_manipulation",
    "goal_spoof_midep":          "goal_manipulation",
    "sensor_dropout":            "sensor_info_loss",
    "sensor_bias":               "sensor_info_loss",
    "contact_dropout":           "sensor_info_loss",
}

# ---------------------------------------------------------------------------
# TAIRO-HX Level 1 (Task Stage) labeling — memo's 6-stage ordered scheme
# (TAIRO-HX.md Section 3). Deterministic, causal-only cascade — see
# scripts/build_level1_labels.py. Approach->Align reuses REACH_THRESHOLD
# (no new constant). Transport->Place is a NEW, PROVISIONAL boundary with no
# prior precedent in this repo -- calibrate against the clean-episode
# dttg-at-arrival distribution before treating as final (decided 2026-07-20).
# ---------------------------------------------------------------------------
LEVEL1_STAGES = [
    "approaching_object",
    "aligning_gripper",
    "grasping",
    "transporting",
    "placing",
    "verifying_completion",
]
LEVEL1_PLACE_RADIUS = 0.10  # m -- PROVISIONAL, see comment above

# ---------------------------------------------------------------------------
# TAIRO-HX Level 5 (Recovery Decision) — memo's 7-decision scheme
# (TAIRO-HX.md Section 3, "Level 5: Recoverability"). Rule-based decision
# logic over Levels 2-4's CHAINED PREDICTIONS (not ground truth, not Level
# 1 — see CLAUDE.md "Level-Chaining Architecture"), not a learned
# classifier. See scripts/build_level5_labels.py.
# ---------------------------------------------------------------------------
LEVEL5_DECISIONS = [
    "continue_sac_her",
    "compensate_for_problem",
    "reconstruct_state",
    "retry_task_stage",
    "restore_trusted_goal",
    "continue_reduced_speed",
    "stop_safely",
]
# Both PROVISIONAL, uncalibrated — proxies for the memo's safe-stop triggers
# (TAIRO-HX.md's untitled section between "Classifier algorithms" and
# "Improved recovery families"). "Loss of both vision and contact" has no
# implementable analog in this sim (no camera/perception channel exists —
# see RECOVERY_V4.md section 2.5's oracle-privilege discussion); folded into
# the low-confidence proxy below rather than invented.
LEVEL5_LOW_CONFIDENCE_THRESH = 0.4   # level3_confidence / level4_confidence
LEVEL5_ABNORMAL_STREAK_THRESH = 2    # consecutive prior abnormal checkpoints

# ---------------------------------------------------------------------------
# Recovery v4-HX: stage-gated expert mixture (mentor-directed, 2026-07-20).
# Multiplies recovery_v4.py's existing per-expert class-probability weights
# by a Level-1-task-stage compatibility mask before mixing -- the one signal
# (Level 1) recovery_v4.py's CCAR never used at all. See
# recovery/recovery_v4_hx.py. Confirmed via sign-off: SOFT gating (down-weight
# by STAGE_EXPERT_SOFT_WEIGHT, not hard-zero) for stages an expert's own
# docstring precondition doesn't cover -- Level 1's own cross-tab shows a
# small (~5.7% of episodes) stage-ordinal regression rate, so a hard zero
# would risk fully silencing a genuinely-needed expert on a Level-1 misfire.
# Keyed by the same EXPERTS dict keys in recovery_v4.py (failure_mode label
# names, not task stages). PROVISIONAL / uncalibrated, same status as
# LEVEL1_PLACE_RADIUS and the Level 5 thresholds above.
STAGE_EXPERT_SOFT_WEIGHT = 0.15

STAGE_EXPERT_COMPAT = {
    # never_reached_object -> approach_expert: navigates toward the object,
    # pointless once contact/grasp is already underway.
    "never_reached_object": {"approaching_object", "aligning_gripper"},
    # reached_but_failed_grasp -> grasp_stabilize_expert: includes
    # "transporting" deliberately -- matches the already-documented
    # causal-proxy tradeoff (CLAUDE.md "Level 1 (Task Stage) Labeling":
    # 30% of reached_but_failed_grasp episodes reach transporting+, kept
    # as-is, Option A) rather than re-litigating that resolved decision.
    "reached_but_failed_grasp": {"aligning_gripper", "grasping", "transporting"},
    # divergent_transport -> transport_expert: its own docstring assumes
    # "the object is already grasped."
    "divergent_transport": {"transporting", "placing", "verifying_completion"},
    # grasped_but_dropped -> regrasp_expert: retreat-and-reapproach only
    # makes sense after a drop, which requires a prior grasp.
    "grasped_but_dropped": {"grasping", "transporting", "placing"},
    # spoofed_goal -> relocalization_expert: goal/object-pose corruption is
    # not stage-bound (active from step 0 for goal_spoof_immediate, or
    # injected mid-episode for goal_spoof_midep) -- all 6 stages eligible,
    # i.e. this expert is never soft-gated.
    "spoofed_goal": set(LEVEL1_STAGES),
}

# ---------------------------------------------------------------------------
# Recovery v4-HX2: Level 4 (attack family) down-weighting, layered on top of
# v4-HX's Level 1 stage gating (mentor-directed, 2026-07-20; see
# recovery/recovery_v4_hx2.py). RECOVERY_V4.md section 2.5 already argues
# action-channel attacks (action_clipping/delay/reversal,
# grip_state_falsification -- config.ATTACK_FAMILY_MAP's "action_actuation"
# family) need secure actuation / command authentication, not better state
# estimation -- so a confident action_actuation prediction down-weights the
# final blend weight rather than trusting the recovery experts more.
# PROVISIONAL / uncalibrated, same status as the Level 5 / stage-gating
# constants above.
LEVEL4_ACTION_ACTUATION_DOWNWEIGHT = 0.3   # multiplier on final blend weight w
LEVEL4_CONFIDENT_THRESH = 0.5              # min level4 max-class-prob to apply it

# ---------------------------------------------------------------------------
# Recovery v4-HX5: fast-attack/slow-release trigger EMA, layered on top of
# v4-HX2's mixture (mentor-directed goal-spoof investigation, 2026-07-22; see
# recovery/recovery_v4_hx5.py). recovery_v4.py's TriggerWeight.update() uses a
# single symmetric EMA (alpha=DEFAULT_ALPHA~=0.05, ~20-step time constant) --
# confirmed via hx4's step logs to take 40-60+ steps to reach even w~=0.3-0.5
# on goal_spoof_immediate/midep, vs. recovery_v3's 3-10 step trigger + full
# (unblended) authority once triggered. RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER
# speeds up the EMA only on the rising (p_fail > ema_pfail) side, leaving the
# original alpha for the falling side unchanged -- so the carefully-tuned
# clean-episode decay behavior (Phase 5b, RECOVERY_V4.md section 2.6) is not
# touched, only how fast the trigger reacts to a NEW sustained failure signal.
# PROVISIONAL / uncalibrated (chosen to bring the ~20-step time constant down
# to ~5 steps, roughly matching v3's trigger speed) -- same status as the
# other Recovery v4-HX* constants above; empirically checked via the standard
# do-no-harm audit for clean-episode side effects, not separately calibrated.
RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER = 4.0

# ---------------------------------------------------------------------------
# Recovery v4-HX6: Level-4-gated version of hx5's fast-attack trigger EMA
# (mentor-directed continuation of the trigger-speed work, 2026-07-22; see
# recovery/recovery_v4_hx6.py). hx5 applied RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER
# globally and showed a soft regression on the grip_state_falsification win;
# hx6 reuses the SAME multiplier value (never shown to be wrong in magnitude,
# only in scope) but only fires it when Level 4 confidently predicts
# perception_state/goal_manipulation (LEVEL4_CONFIDENT_THRESH, reused from
# the hx2 down-weight gate) -- the families the ramp-lag investigation
# actually targeted. No new numeric constant needed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym          # noqa: F401
    import gymnasium_robotics        # noqa: F401
    GYM_AVAILABLE = True
except Exception as _gym_err:
    GYM_AVAILABLE = False
    print(f"[config] Gymnasium Robotics not available: {_gym_err!r}")

try:
    from stable_baselines3 import SAC                           # noqa: F401
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer  # noqa: F401
    SB3_AVAILABLE = True
except Exception as _sb3_err:
    SB3_AVAILABLE = False
    print(f"[config] Stable-Baselines3 not available: {_sb3_err!r}")
