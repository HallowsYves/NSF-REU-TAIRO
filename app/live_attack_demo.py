"""
Live interactive demo: SAC+HER (clean_2M) on FetchPickAndPlace-v4 under
adversarial attack conditions, rendered step-by-step in a Streamlit page.

Run with:
    conda activate reu_robotics
    streamlit run app/live_attack_demo.py

v1 scope: live render + attack controls (condition + magnitude/toggle) +
step counter + end-of-episode success/fail readout.

v2 scope: a live metrics panel (distance-to-goal, C4 jerk/safety
indicators) and a classifier/recovery overlay (online failure-mode
classifier's live p_fail, Recovery v4's trigger/blend-weight state). The
overlay was telemetry only -- it did not wire Recovery v4's blended action
into the executed action.

v3 (this file): a recovery-comparison demo. Two synchronized episodes run
side by side from the same seed and the identical attack instance -- the
raw SAC+HER policy (left) and SAC+HER + Recovery v4 CCAR (right, using
recovery.recovery_v4.recovery_step to actually blend the action, not just
display telemetry) -- plus an end-of-episode trustworthiness section that
shows both this run's real outcome and the committed, statistically
validated C1-C5 benchmark numbers for the selected condition.
"""

import os
import pickle
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import SAC

import config
from envs.fetchpickandplace_env import make_env, distance_to_goal
from policies.sac_her_policy import SACHerPolicy
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack
from evaluation.episode_runner import _pnp_spatial_fields
from recovery.recovery_v4 import (
    TriggerWeight, ExpertState, recovery_step, EPSILON as RECOVERY_V4_EPSILON,
)
from attacks.sensor_attacks import apply_sensor_bias, shift_target, apply_object_pose_spoof
from app.sim_worker import SimWorker, SimWorkerCrashed

# Seed-fixed classifier/calibration artifacts (see CLAUDE.md section 10 --
# config.CLASSIFIER_DIR's default does not contain these; the seedfix dir
# must be named explicitly, same as run_multiseed_sweep.py's
# --recovery-v4-classifier-dir convention).
CLASSIFIER_SEEDFIX_DIR = "results/classifier_seedfix"

# Committed Tier-1 CCAR benchmark (5 seeds x 30 episodes/seed, clean_2M
# checkpoint) -- already has C1-C5 computed per (method, condition,
# attack_level) via evaluation/metrics.py's add_trustworthiness_scores().
# Used to ground the live comparison in the real, statistically validated
# numbers rather than inventing a single-episode approximation of what is,
# in evaluation/metrics.py, always a rate-based metric across many episodes.
BENCHMARK_SUMMARY_PATH = "results/data_recovery_v4/phase6_summary_sac_her_pickandplace_clean_2M.csv"

# A concrete, verified example of Recovery v4 flipping a failed episode
# into a success -- taken directly from the committed episode CSV
# (results/data_recovery_v4/episode_results_sac_her_pickandplace_clean_2M.csv,
# sac_her vs sac_her_recovery_v4, reset_seed = 100*seed + episode_in_seed
# matching evaluation/episode_runner.py's convention). action_delay is one
# of only two conditions with a clean net-positive lift and zero
# down-flips in that grid (the other is action_clipping); most other
# conditions/seeds show no visible per-episode difference, since Tier 1
# CCAR's real aggregate benefit on this checkpoint is small (<2pp best
# case; grip_state_falsification is a net *regression*). See
# RECOVERY_V4.md for the full picture.
#
# This CSV-derived value is safe to use again as of the mj_copyData fix in
# sim_worker.py's _render_preserving_physics -- until that fix, SimWorker
# calling env.render() every step measurably perturbed MuJoCo's physics
# (confirmed root cause: gymnasium_robotics's FetchEnv._render_callback
# calls mj_forward() to reposition a cosmetic goal-marker site, which as a
# side effect re-runs the contact solver and changes data.qacc_warmstart),
# so a seed's outcome in the headless batch eval did not reliably predict
# its outcome in this rendered demo. Re-verified through the actual
# SimWorker subprocess after the fix: action_delay/seed=6 now reproduces
# exactly clean=FAIL, recovery=SUCCESS, matching the CSV.
SUGGESTED_EXAMPLE = {"condition": "action_delay", "seed": 6}

# FetchPickAndPlace-v4's action space is the standard Box(-1, 1, (4,))
# (same convention already hardcoded in attacks/action_attacks.py's
# clipping default). SimWorker's subprocess IPC protocol doesn't expose
# env.action_space, so this is hardcoded to match rather than queried.
ACTION_LOW, ACTION_HIGH = -1.0, 1.0

# Conditions whose ATTACK_LEVELS/TRAIN_ATTACK_RANGES magnitude is a fixed 0.0
# placeholder — i.e. structural/binary attacks, not magnitude-scaled ones.
BINARY_CONDITIONS = {
    "sensor_dropout",
    "action_delay",
    "action_reversal",
    "grip_state_falsification",
    "contact_dropout",
}

# Poll rate of the auto-play fragment (see render_fragment below).
#
# MUST stay well above ~0.1s. Empirically (via a real browser, not
# streamlit's AppTest harness, which doesn't exercise this): a run_every
# this fast races with a normal full-page rerun (e.g. clicking Step/Reset,
# which live outside this fragment) -- when a fragment auto-tick and a
# full rerun land close together, the fragment's next full-rerun-triggered
# execution silently writes into a stale container snapshot and the whole
# render pane goes blank permanently (confirmed reproducible at 0.05s
# across repeated clean-server trials; confirmed NOT reproducible at
# 0.1s-3.0s). 0.25s (4 Hz) keeps solid margin below that boundary.
RENDER_TICK_SECONDS = 0.25

# Playback speed is fixed, not user-adjustable -- RENDER_TICK_SECONDS (4 Hz)
# is already the practical ceiling (see the docstring above).
PLAYBACK_FPS = 4


@st.cache_resource
def load_policy() -> SACHerPolicy:
    load_env = make_env(seed=0)
    model = SAC.load(config.MODEL_PATH_PICKANDPLACE_2M, env=load_env)
    load_env.close()
    return SACHerPolicy(model)


@st.cache_resource
def load_recovery_v4_artifacts():
    """Online failure-mode classifier + Recovery v4 trigger calibration.

    Loaded once and reused across the whole app lifetime (same rationale as
    load_policy) -- both are read-only, checkpoint-scoped to clean_2M (the
    only model this demo runs), and safe to share across sessions and
    across both panes.
    """
    classifier_path = os.path.join(CLASSIFIER_SEEDFIX_DIR, "online_failure_classifier.pkl")
    calibration_path = os.path.join(CLASSIFIER_SEEDFIX_DIR, "recovery_v4_trigger_calibration.pkl")
    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)
    with open(calibration_path, "rb") as f:
        calibration = pickle.load(f)
    return classifier, calibration


@st.cache_resource
def load_benchmark_summary() -> pd.DataFrame:
    return pd.read_csv(BENCHMARK_SUMMARY_PATH)


def lookup_benchmark_row(df: pd.DataFrame, method: str, condition: str, attack_level: float):
    """Nearest-attack_level row for (method, condition), or None if the
    condition isn't in the committed benchmark at all. Nearest rather than
    exact match since the live magnitude slider can, in principle, be
    moved off the sweep's fixed calibrated value.
    """
    sub = df[(df["method"] == method) & (df["condition"] == condition)]
    if sub.empty:
        return None
    idx = (sub["attack_level"] - attack_level).abs().idxmin()
    return sub.loc[idx]


class PaneState:
    """Per-pane (clean vs recovery) episode state. Two instances live in
    st.session_state (pane_clean, pane_recovery) and are mutated in place
    across reruns/fragment ticks -- same pattern the v1/v2 single-pane
    version used for its TriggerWeight instance.
    """

    def __init__(self, name: str, use_recovery: bool) -> None:
        self.name = name
        self.use_recovery = use_recovery
        self.worker = None
        self.obs = None
        self.last_frame = None
        self.step_count = 0
        self.bias_vector = None
        self.goal_offset = None
        self.object_pose_offset = None
        self.previous_action = None
        self.prev_executed = None
        self.step_history = []
        self.done = False
        self.success = None
        self.worker_error = None
        self.distance_to_goal_now = None
        self.arm_jerk = 0.0
        self.grip_jerk = 0.0
        self.safety_violation_step = 0.0
        # Recovery-pane-only fields -- left at these harmless defaults for
        # the clean pane, which never populates them.
        self.trigger = None
        self.expert_state = None
        self.class_probs = {}
        self.p_fail = None
        self.ema_pfail = 0.0
        self.recovery_weight = 0.0
        self.first_recovery_step = float("nan")


def precompute_shared_attack_vectors(condition: str, obs, attack_level: float, seed: Optional[int]):
    """Draw the per-episode attack vector ONCE, so both panes are attacked
    identically instead of each independently self-sampling.

    apply_sensor_attack (evaluation/attack_dispatch.py) never threads a
    seed down to the low-level attack functions, so two independent calls
    -- one per pane -- would sample DIFFERENT bias_vector/goal_offset/
    object_pose_offset even with the same episode seed, confounding the
    clean-vs-recovery comparison with unrelated randomness. Fixed here by
    calling the real attack functions directly (not apply_sensor_attack)
    with an explicit seed, once, and reusing the result for both panes --
    no changes to attacks/sensor_attacks.py or evaluation/attack_dispatch.py
    (both explicitly documented as shared and already correct).

    For goal_spoof_midep specifically: shift_step=None is passed here (not
    GOAL_SPOOF_MIDEP_STEP) purely to force immediate sampling so a value
    exists to seed both panes with. This does NOT skip the real onset
    gating -- apply_sensor_attack's own per-step call (in step_pane) still
    passes the real shift_step, and its return-value threading
    (`if new_offset is not None: goal_offset = new_offset`) already
    preserves a pre-seeded non-None goal_offset across the pre-step-60
    None-returns untouched, so the mid-episode activation timing is
    unaffected.
    """
    bias_vector = goal_offset = object_pose_offset = None
    if condition == "sensor_bias":
        _, bias_vector = apply_sensor_bias(obs, magnitude=attack_level, bias_vector=None, seed=seed)
    elif condition in ("goal_spoof_immediate", "goal_spoof_midep"):
        _, goal_offset = shift_target(
            obs, shift_scale=attack_level, step=0, shift_step=None, goal_offset=None, seed=seed,
        )
    elif condition == "object_pose_spoof":
        _, object_pose_offset = apply_object_pose_spoof(
            obs, magnitude=attack_level, offset=None, seed=seed,
        )
    return bias_vector, goal_offset, object_pose_offset


# Conditions whose per-episode attack vector must be shared between panes
# (see precompute_shared_attack_vectors), mapped to the PaneState attribute
# that holds it.
SHARED_VECTOR_CONDITIONS = {
    "sensor_bias": "bias_vector",
    "goal_spoof_immediate": "goal_offset",
    "goal_spoof_midep": "goal_offset",
    "object_pose_spoof": "object_pose_offset",
}


def ensure_shared_attack_vectors(condition: str, attack_level: float) -> None:
    """Re-seed both panes' shared attack vector if the user switched to a
    vector-sampling condition mid-episode (no Reset) and it hasn't been
    seeded for this condition yet.

    reset_episode() seeds the shared vector once per episode, but changing
    the condition dropdown mid-episode without clicking Reset bypasses it
    entirely (a documented "no Reset required" simplification carried over
    from v1) -- without this check, each pane would independently
    self-sample a DIFFERENT vector on its next apply_sensor_attack call,
    breaking the "identical attack instance" guarantee this comparison
    depends on. Confirmed as a real gap via a standalone check before this
    fix: switching clean -> sensor_bias mid-episode without Reset gave the
    two panes different bias vectors.

    No seed is threaded through here (unlike reset_episode, which uses the
    episode seed for reproducibility) -- a live mid-episode switch has no
    natural seed to tie to, and the only real requirement is that both
    panes agree, not that this correction is itself reproducible run to
    run.
    """
    attr = SHARED_VECTOR_CONDITIONS.get(condition)
    if attr is None:
        return
    ss = st.session_state
    pc, pr = ss.pane_clean, ss.pane_recovery
    if getattr(pc, attr) is not None or getattr(pr, attr) is not None:
        return  # already seeded for this condition, by Reset or a prior call

    seed_obs = pc.obs if pc.worker_error is None else pr.obs
    if seed_obs is None:
        return
    bias_vector, goal_offset, object_pose_offset = precompute_shared_attack_vectors(
        condition, seed_obs, attack_level, seed=None,
    )
    for pane in (pc, pr):
        pane.bias_vector = bias_vector
        pane.goal_offset = goal_offset
        pane.object_pose_offset = object_pose_offset


def init_state() -> None:
    if "initialized" in st.session_state:
        return
    ss = st.session_state
    ss.initialized = True
    ss.pane_clean = PaneState(name="clean", use_recovery=False)
    ss.pane_recovery = PaneState(name="recovery", use_recovery=True)
    ss.running = False
    ss.last_step_time = 0.0
    reset_episode(seed=0, condition="clean", attack_level=0.0)


def reset_episode(seed: int, condition: str, attack_level: float) -> None:
    ss = st.session_state
    for pane in (ss.pane_clean, ss.pane_recovery):
        try:
            if pane.worker is None:
                pane.worker = SimWorker(seed=seed, name=pane.name)
                pane.obs, pane.last_frame = pane.worker.obs, pane.worker.frame
            else:
                pane.obs, pane.last_frame = pane.worker.reset(seed)
            pane.worker_error = None
        except SimWorkerCrashed as e:
            pane.worker = None
            pane.worker_error = str(e)
            pane.done = True
            continue

        pane.step_count = 0
        pane.previous_action = None
        pane.prev_executed = None
        pane.step_history = []
        pane.done = False
        pane.success = None
        pane.distance_to_goal_now = distance_to_goal(pane.obs)
        pane.arm_jerk = 0.0
        pane.grip_jerk = 0.0
        pane.safety_violation_step = 0.0

        if pane.use_recovery:
            _classifier, calibration = load_recovery_v4_artifacts()
            pane.trigger = TriggerWeight(clean_pfail_p95=calibration["clean_2M"])
            pane.expert_state = ExpertState()
            pane.class_probs = {}
            pane.p_fail = None
            pane.ema_pfail = 0.0
            pane.recovery_weight = 0.0
            pane.first_recovery_step = float("nan")

    ss.running = False
    ss.last_step_time = 0.0

    # Shared attack instance -- see precompute_shared_attack_vectors's
    # docstring. Falls back to whichever pane's worker is alive if one
    # crashed at reset; both started from the same seed so either obs
    # works as the sampling basis.
    seed_obs = None
    if ss.pane_clean.worker_error is None:
        seed_obs = ss.pane_clean.obs
    elif ss.pane_recovery.worker_error is None:
        seed_obs = ss.pane_recovery.obs

    bias_vector = goal_offset = object_pose_offset = None
    if seed_obs is not None:
        bias_vector, goal_offset, object_pose_offset = precompute_shared_attack_vectors(
            condition, seed_obs, attack_level, seed
        )
    for pane in (ss.pane_clean, ss.pane_recovery):
        pane.bias_vector = bias_vector
        pane.goal_offset = goal_offset
        pane.object_pose_offset = object_pose_offset


def step_pane(pane: PaneState, condition: str, attack_level: float) -> None:
    """One step for a single pane. Shared by both the clean and recovery
    panes -- the only behavioral difference is the recovery-blend block,
    gated on pane.use_recovery. Mirrors evaluation/episode_runner.py's
    exact ordering: sensor attack -> policy -> action attack -> recovery
    blend (if applicable) -> C4 jerk on the post-blend action -> env.step.
    """
    if pane.done or pane.worker is None:
        return

    policy_obs, pane.bias_vector, pane.goal_offset, pane.object_pose_offset = apply_sensor_attack(
        condition, pane.obs, pane.step_count, pane.bias_vector, pane.goal_offset,
        attack_level=attack_level, object_pose_offset=pane.object_pose_offset,
    )

    intended_action = np.asarray(st.session_state.policy(None, policy_obs), dtype=np.float32)

    executed_action = apply_action_attack(
        condition, intended_action, pane.previous_action, attack_level=attack_level,
    )

    pane.previous_action = (
        intended_action.copy() if condition == "action_delay" else executed_action.copy()
    )

    # -- Recovery v4 blend (recovery pane only). Uses raw obs (pane.obs,
    # pre-sensor-attack) and step_history built from steps [0, step_count-1]
    # only -- recovery_step's own step-0 guard handles the first step. ------
    if pane.use_recovery:
        classifier, _calibration = load_recovery_v4_artifacts()
        step_history_df = pd.DataFrame(pane.step_history)
        executed_action, pane.class_probs, pane.recovery_weight = recovery_step(
            policy_action=executed_action,
            obs=pane.obs,
            step_history_df=step_history_df,
            classifier_artifact=classifier,
            trigger=pane.trigger,
            expert_state=pane.expert_state,
            step=pane.step_count,
        )
        executed_action = np.clip(executed_action, ACTION_LOW, ACTION_HIGH).astype(np.float32)
        pane.ema_pfail = pane.trigger.ema_pfail
        pane.p_fail = 1.0 - pane.class_probs["success"] if pane.class_probs else None
        if pane.recovery_weight >= RECOVERY_V4_EPSILON and np.isnan(pane.first_recovery_step):
            pane.first_recovery_step = float(pane.step_count)

    # -- C4 jerk metric (per-channel split, matches evaluation/episode_runner.py
    # exactly -- always the last *executed* action (post-recovery-blend for
    # the recovery pane), never previous_action, per CLAUDE.md section 8) ---
    if pane.prev_executed is not None:
        pane.arm_jerk = float(np.linalg.norm(executed_action[:3] - pane.prev_executed[:3]))
        pane.grip_jerk = float(abs(executed_action[3] - pane.prev_executed[3]))
        pane.safety_violation_step = float(
            pane.arm_jerk > config.SAFETY_ARM_JERK_THRESHOLD
            or pane.grip_jerk > config.SAFETY_GRIPPER_JERK_THRESHOLD
        )
    else:
        pane.arm_jerk = 0.0
        pane.grip_jerk = 0.0
        pane.safety_violation_step = 0.0

    pane.prev_executed = executed_action.copy()

    try:
        pane.obs, reward, terminated, truncated, info, pane.last_frame = pane.worker.step(
            executed_action
        )
    except SimWorkerCrashed as e:
        pane.worker = None
        pane.worker_error = str(e)
        pane.done = True
        return

    pane.distance_to_goal_now = distance_to_goal(pane.obs)
    is_success = float(info.get("is_success", 0.0))
    spatial = _pnp_spatial_fields(pane.obs, pane.goal_offset)
    pane.step_history.append({
        "timestep": pane.step_count,
        "reward": float(reward),
        "is_success": is_success,
        "action_norm": float(np.linalg.norm(executed_action)),
        "intended_action_norm": float(np.linalg.norm(intended_action)),
        "safety_violation": pane.safety_violation_step,
        **spatial,
    })

    pane.step_count += 1

    if terminated or truncated:
        pane.done = True
        pane.success = bool(is_success)


def render_pane(pane: PaneState, title: str, render_slot, metrics_slot, show_classifier: bool) -> None:
    with render_slot:
        st.markdown(f"**{title}**")
        if pane.worker_error:
            st.error(f"Render subprocess crashed: {pane.worker_error}")
        else:
            # Fixed display width, well under the source 960x960 render --
            # shrinks the on-page footprint (two panes side by side) without
            # touching render quality (the browser downsamples the frame).
            st.image(pane.last_frame, channels="rgb", width=420)
            st.write(f"Step {pane.step_count} / {config.MAX_EPISODE_STEPS_PICKANDPLACE}")
            if pane.done:
                if pane.success:
                    st.success("SUCCESS")
                else:
                    st.error("FAILURE")

    if pane.worker_error:
        return

    with metrics_slot:
        st.metric("Distance to goal", f"{pane.distance_to_goal_now:.3f} m")
        jcol1, jcol2 = st.columns(2)
        jcol1.metric("Arm jerk", f"{pane.arm_jerk:.2f}")
        jcol2.metric("Grip jerk", f"{pane.grip_jerk:.2f}")
        if pane.safety_violation_step:
            st.warning("C4 safety violation this step")
        else:
            st.caption("No C4 safety violation this step")

        if show_classifier:
            st.divider()
            st.markdown("**Classifier / Recovery v4**")
            if pane.class_probs:
                predicted_label = max(pane.class_probs, key=pane.class_probs.get)
                st.write(f"Predicted: **{predicted_label}** (p={pane.class_probs[predicted_label]:.2f})")
                st.progress(min(max(pane.p_fail, 0.0), 1.0), text=f"p_fail = {pane.p_fail:.3f}")
                st.caption(f"EMA(p_fail) = {pane.ema_pfail:.3f}")
                st.progress(
                    min(max(pane.recovery_weight, 0.0), 1.0),
                    text=f"Recovery v4 blend weight w = {pane.recovery_weight:.3f}",
                )
            else:
                st.caption(
                    "Classifier telemetry needs 1 step of history — available "
                    "from step 2 onward."
                )


def render_trustworthiness_section(slot, condition: str, attack_level: float) -> None:
    ss = st.session_state
    pc, pr = ss.pane_clean, ss.pane_recovery

    with slot:
        if pc.worker_error or pr.worker_error:
            # step_pane() sets pane.done = True on a SimWorkerCrashed too
            # (it's a terminal state for that pane), which would otherwise
            # satisfy the "both done" check below and render this section
            # as if the episode legitimately finished -- pc.success/
            # pr.success are never set on a crash, so the comparison below
            # would silently compare against None. render_pane() already
            # shows the crash itself; this section just stays hidden.
            st.caption("Trustworthiness comparison unavailable — a render subprocess crashed this run.")
            return
        if not (pc.done and pr.done):
            st.caption("Trustworthiness comparison appears here once both episodes finish.")
            return

        st.subheader("Trustworthiness — this run")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Clean policy**")
            st.write("Success" if pc.success else "Failure")
            st.caption(f"Final distance to goal: {pc.distance_to_goal_now:.3f} m")
            n_violations = int(sum(row["safety_violation"] for row in pc.step_history))
            st.caption(f"C4 safety violations: {n_violations} step(s)")
        with c2:
            st.markdown("**Policy + Recovery v4**")
            st.write("Success" if pr.success else "Failure")
            st.caption(f"Final distance to goal: {pr.distance_to_goal_now:.3f} m")
            n_violations = int(sum(row["safety_violation"] for row in pr.step_history))
            st.caption(f"C4 safety violations: {n_violations} step(s)")
            if not np.isnan(pr.first_recovery_step):
                st.caption(f"Recovery first triggered at step {int(pr.first_recovery_step)}")
            else:
                st.caption("Recovery never triggered above threshold this episode")

        if pc.success and not pr.success:
            st.warning(
                "Recovery made this episode worse — the raw policy succeeded, "
                "recovery did not. A real, documented possibility (see "
                "RECOVERY_V4.md's weak spots), not a bug."
            )
        elif not pc.success and pr.success:
            st.success(
                "Recovery saved this episode — the raw policy failed, "
                "recovery succeeded."
            )
        elif pc.success and pr.success:
            st.info("Both succeeded — no visible difference this episode.")
        else:
            st.info("Both failed — recovery did not save this episode.")

        st.divider()
        st.markdown("**How this compares to the full benchmark**")
        st.caption(
            "From the committed 5-seed × 30-episode-per-seed evaluation "
            "(results/data_recovery_v4), not just this one run."
        )
        bdf = load_benchmark_summary()
        base_row = lookup_benchmark_row(bdf, "sac_her", condition, attack_level)
        rec_row = lookup_benchmark_row(bdf, "sac_her_recovery_v4", condition, attack_level)
        if base_row is None or rec_row is None:
            st.caption("No committed benchmark row for this condition/magnitude.")
        else:
            table = pd.DataFrame(
                {
                    "sac_her": [
                        base_row["reliability_score"], base_row["safety_score"],
                        base_row["recovery_score"], base_row["trustworthiness_score_weighted"],
                    ],
                    "sac_her_recovery_v4": [
                        rec_row["reliability_score"], rec_row["safety_score"],
                        rec_row["recovery_score"], rec_row["trustworthiness_score_weighted"],
                    ],
                },
                index=["C1 reliability", "C4 safety", "C5 recovery", "Weighted trustworthiness"],
            )
            st.dataframe(table.style.format("{:.3f}"))


@st.fragment(run_every=RENDER_TICK_SECONDS)
def render_fragment(
    condition: str, attack_level: float, fps: int,
    render_slot_clean, metrics_slot_clean,
    render_slot_recovery, metrics_slot_recovery,
    trust_slot,
) -> None:
    """Live render + telemetry for both panes. Reruns on its own timer (see
    RENDER_TICK_SECONDS) without rerunning the rest of the page. All slots
    are `st.container()`s created once in main() so this single fragment
    updates every location on every tick, keeping both panes in lockstep.
    """
    ss = st.session_state
    pc, pr = ss.pane_clean, ss.pane_recovery

    if ss.running and not (pc.done and pr.done):
        now = time.monotonic()
        if now - ss.last_step_time >= 1.0 / fps:
            ensure_shared_attack_vectors(condition, attack_level)
            if not pc.done:
                step_pane(pc, condition, attack_level)
            if not pr.done:
                step_pane(pr, condition, attack_level)
            ss.last_step_time = now
            if pc.done and pr.done:
                ss.running = False

    # A fragment auto-tick can occasionally land in the same instant as a
    # full-page rerun triggered by a button click (Reset/Start/etc, which
    # live outside this fragment) -- the same race RENDER_TICK_SECONDS is
    # calibrated against (see its docstring above). With five containers
    # now written per tick (two render slots, two metrics slots, one
    # trustworthiness slot) instead of v1/v2's two, Streamlit's container
    # reservation check can raise StreamlitAPIException on the losing
    # side of that race instead of just silently skipping a frame. Caught
    # here so a rare collision degrades to "this tick renders nothing"
    # (the next tick, ~0.25s later, always succeeds) rather than an
    # uncaught exception.
    try:
        render_pane(pc, "Clean policy (no recovery)", render_slot_clean, metrics_slot_clean, show_classifier=False)
        render_pane(pr, "Policy + Recovery v4", render_slot_recovery, metrics_slot_recovery, show_classifier=True)
        render_trustworthiness_section(trust_slot, condition, attack_level)
    except st.errors.StreamlitAPIException:
        pass


def main() -> None:
    st.set_page_config(page_title="TAIRO Live Attack Demo", layout="wide")
    init_state()
    st.session_state.policy = load_policy()

    st.title("SAC+HER (clean_2M) — Recovery Comparison Demo")
    st.caption(
        "Two synchronized episodes, same seed and identical attack instance: "
        "the raw SAC+HER policy on the left, SAC+HER + Recovery v4 (CCAR) on "
        "the right. Try condition "
        f"'{SUGGESTED_EXAMPLE['condition']}' with seed {SUGGESTED_EXAMPLE['seed']} "
        "(the default seed — just switch the condition and hit Reset) for a "
        "verified example of recovery saving an otherwise-failed episode — "
        "most seeds show no difference, since Recovery v4's real aggregate "
        "benefit on this checkpoint is a small lift on a couple of "
        "conditions, not a dramatic per-episode effect."
    )

    st.subheader("Attack controls")
    ctrl_cols = st.columns(2)
    with ctrl_cols[0]:
        condition = st.selectbox("Attack condition", config.ALL_CONDITIONS, key="condition")

        attack_level = 0.0
        if condition == "clean":
            st.caption("No attack active.")
        elif condition in BINARY_CONDITIONS:
            active = st.checkbox("Attack active", value=True, key=f"toggle_{condition}")
            if not active:
                condition = "clean"
        else:
            low, high = config.TRAIN_ATTACK_RANGES[condition]
            default = config.ATTACK_LEVELS[condition]
            attack_level = st.slider(
                "Magnitude", min_value=float(low), max_value=float(high),
                value=float(default), key=f"mag_{condition}",
            )
    with ctrl_cols[1]:
        seed = st.number_input("Episode seed", value=0, step=1, key="seed_input")
        st.caption(
            "Changing condition/magnitude mid-episode takes effect from the "
            "next step onward (no Reset required) — a known v1 simplification, "
            "not a bug. Both panes always receive an identical attack instance "
            "for a fair comparison."
        )

    st.subheader("Playback")
    pb_col, _spacer = st.columns([1, 3])
    with pb_col:
        both_done = st.session_state.pane_clean.done and st.session_state.pane_recovery.done
        prow1 = st.columns(2)
        prow2 = st.columns(2)
        if prow1[0].button("Start", disabled=both_done, width="stretch"):
            st.session_state.running = True
        if prow1[1].button("Pause", width="stretch"):
            st.session_state.running = False
        if prow2[0].button("Step", disabled=both_done, width="stretch"):
            ensure_shared_attack_vectors(condition, attack_level)
            if not st.session_state.pane_clean.done:
                step_pane(st.session_state.pane_clean, condition, attack_level)
            if not st.session_state.pane_recovery.done:
                step_pane(st.session_state.pane_recovery, condition, attack_level)
        if prow2[1].button("Reset", width="stretch"):
            reset_episode(int(seed), condition, attack_level)

    st.divider()

    pane_cols = st.columns(2)
    with pane_cols[0]:
        render_slot_clean = st.container()
        metrics_slot_clean = st.container()
    with pane_cols[1]:
        render_slot_recovery = st.container()
        metrics_slot_recovery = st.container()

    trust_slot = st.container()

    render_fragment(
        condition, attack_level, PLAYBACK_FPS,
        render_slot_clean, metrics_slot_clean,
        render_slot_recovery, metrics_slot_recovery,
        trust_slot,
    )


if __name__ == "__main__":
    main()
