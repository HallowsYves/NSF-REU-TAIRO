"""
Live interactive demo: SAC+HER (clean_2M) on FetchPickAndPlace-v4 under
adversarial attack conditions, rendered step-by-step in a Streamlit page.

Run with:
    conda activate reu_robotics
    streamlit run app/live_attack_demo.py

v1 scope: live render + attack controls (condition + magnitude/toggle) +
step counter + end-of-episode success/fail readout.

v2 (this file): a live metrics panel (distance-to-goal, C4 jerk/safety
indicators) and a classifier/recovery overlay (online failure-mode
classifier's live p_fail, Recovery v4's trigger/blend-weight state).
The overlay is telemetry only -- it does NOT wire Recovery v4's blended
action into the executed action; the demo always executes the (possibly
attacked) SAC+HER policy action, same as v1.
"""

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import SAC

import config
from envs.fetchpickandplace_env import make_env, distance_to_goal
from policies.sac_her_policy import SACHerPolicy
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack
from evaluation.causal_features import build_causal_features_online
from evaluation.episode_runner import _pnp_spatial_fields
from recovery.recovery_v4 import TriggerWeight, get_class_probs
from app.sim_worker import SimWorker, SimWorkerCrashed

# Seed-fixed classifier/calibration artifacts (see CLAUDE.md section 10 --
# config.CLASSIFIER_DIR's default does not contain these; the seedfix dir
# must be named explicitly, same as run_multiseed_sweep.py's
# --recovery-v4-classifier-dir convention).
CLASSIFIER_SEEDFIX_DIR = "results/classifier_seedfix"

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
# 0.1s-3.0s). 0.25s (4 Hz) keeps solid margin below that boundary. The FPS
# slider's max is capped to match -- stepping faster than one tick allows
# is impossible regardless of the slider, so the slider must not promise
# a rate this mechanism can't sustain.
RENDER_TICK_SECONDS = 0.25

# Playback speed is fixed, not user-adjustable -- RENDER_TICK_SECONDS (4 Hz)
# is already the practical ceiling (see the docstring above), so a slider
# that only ever usefully sits at its max was removed in favor of a constant.
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
    only model this demo runs), and safe to share across sessions.
    """
    classifier_path = os.path.join(CLASSIFIER_SEEDFIX_DIR, "online_failure_classifier.pkl")
    calibration_path = os.path.join(CLASSIFIER_SEEDFIX_DIR, "recovery_v4_trigger_calibration.pkl")
    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)
    with open(calibration_path, "rb") as f:
        calibration = pickle.load(f)
    return classifier, calibration


def init_state() -> None:
    if "initialized" in st.session_state:
        return
    ss = st.session_state
    ss.initialized = True
    ss.worker = None
    ss.obs = None
    ss.step_count = 0
    ss.bias_vector = None
    ss.goal_offset = None
    ss.object_pose_offset = None
    ss.previous_action = None
    ss.running = False
    ss.done = False
    ss.success = None
    ss.last_frame = None
    ss.last_step_time = 0.0
    ss.worker_error = None
    ss.prev_executed = None
    ss.step_history = []
    ss.trigger = None
    ss.distance_to_goal_now = None
    ss.arm_jerk = 0.0
    ss.grip_jerk = 0.0
    ss.safety_violation_step = 0.0
    ss.class_probs = {}
    ss.p_fail = None
    ss.ema_pfail = 0.0
    ss.recovery_weight = 0.0
    reset_episode(seed=0)


def reset_episode(seed: int) -> None:
    ss = st.session_state
    try:
        if ss.get("worker") is None:
            ss.worker = SimWorker(seed=seed)
            ss.obs, ss.last_frame = ss.worker.obs, ss.worker.frame
        else:
            ss.obs, ss.last_frame = ss.worker.reset(seed)
    except SimWorkerCrashed as e:
        ss.worker = None
        ss.worker_error = str(e)
        ss.done = True
        ss.running = False
        return
    ss.step_count = 0
    ss.bias_vector = None
    ss.goal_offset = None
    ss.object_pose_offset = None
    ss.previous_action = None
    ss.done = False
    ss.success = None
    ss.running = False
    ss.last_step_time = 0.0
    ss.worker_error = None
    ss.prev_executed = None
    ss.step_history = []
    _classifier, calibration = load_recovery_v4_artifacts()
    ss.trigger = TriggerWeight(clean_pfail_p95=calibration["clean_2M"])
    ss.distance_to_goal_now = distance_to_goal(ss.obs)
    ss.arm_jerk = 0.0
    ss.grip_jerk = 0.0
    ss.safety_violation_step = 0.0
    ss.class_probs = {}
    ss.p_fail = None
    ss.ema_pfail = 0.0
    ss.recovery_weight = 0.0


def take_step(condition: str, attack_level: float) -> None:
    ss = st.session_state
    if ss.done:
        return

    policy_obs, ss.bias_vector, ss.goal_offset, ss.object_pose_offset = apply_sensor_attack(
        condition, ss.obs, ss.step_count, ss.bias_vector, ss.goal_offset,
        attack_level=attack_level, object_pose_offset=ss.object_pose_offset,
    )

    intended_action = np.asarray(ss.policy(None, policy_obs), dtype=np.float32)

    executed_action = apply_action_attack(
        condition, intended_action, ss.previous_action, attack_level=attack_level,
    )

    ss.previous_action = (
        intended_action.copy() if condition == "action_delay" else executed_action.copy()
    )

    # -- C4 jerk metric (per-channel split, matches evaluation/episode_runner.py
    # exactly -- always the last *executed* action, never previous_action,
    # per CLAUDE.md section 8) -----------------------------------------------
    if ss.prev_executed is not None:
        ss.arm_jerk = float(np.linalg.norm(executed_action[:3] - ss.prev_executed[:3]))
        ss.grip_jerk = float(abs(executed_action[3] - ss.prev_executed[3]))
        ss.safety_violation_step = float(
            ss.arm_jerk > config.SAFETY_ARM_JERK_THRESHOLD
            or ss.grip_jerk > config.SAFETY_GRIPPER_JERK_THRESHOLD
        )
    else:
        ss.arm_jerk = 0.0
        ss.grip_jerk = 0.0
        ss.safety_violation_step = 0.0

    # -- Recovery v4 telemetry (display only -- does NOT alter executed_action).
    # Uses raw obs (ss.obs, pre-sensor-attack) and step_history built from
    # steps [0, step_count-1] only -- same causal discipline as
    # recovery_v4.recovery_step's own step-0 guard. -------------------------
    classifier, _calibration = load_recovery_v4_artifacts()
    if ss.step_count == 0 or len(ss.step_history) == 0:
        ss.class_probs = {}
        ss.p_fail = None
        ss.recovery_weight = 0.0
    else:
        history_df = pd.DataFrame(ss.step_history)
        feature_vec = build_causal_features_online(history_df)
        x = np.array([[feature_vec[c] for c in classifier["feature_cols"]]])
        ss.class_probs = get_class_probs(classifier["model"], x)
        ss.recovery_weight = ss.trigger.update(ss.class_probs["success"])
        ss.p_fail = 1.0 - ss.class_probs["success"]
    ss.ema_pfail = ss.trigger.ema_pfail

    ss.prev_executed = executed_action.copy()

    try:
        ss.obs, reward, terminated, truncated, info, ss.last_frame = ss.worker.step(
            executed_action
        )
    except SimWorkerCrashed as e:
        ss.worker = None
        ss.worker_error = str(e)
        ss.done = True
        ss.running = False
        return

    ss.distance_to_goal_now = distance_to_goal(ss.obs)
    is_success = float(info.get("is_success", 0.0))
    spatial = _pnp_spatial_fields(ss.obs, ss.goal_offset)
    ss.step_history.append({
        "timestep": ss.step_count,
        "reward": float(reward),
        "is_success": is_success,
        "action_norm": float(np.linalg.norm(executed_action)),
        "intended_action_norm": float(np.linalg.norm(intended_action)),
        "safety_violation": ss.safety_violation_step,
        **spatial,
    })

    ss.step_count += 1

    if terminated or truncated:
        ss.done = True
        ss.running = False
        ss.success = bool(is_success)


@st.fragment(run_every=RENDER_TICK_SECONDS)
def render_fragment(condition: str, attack_level: float, fps: int, render_slot, metrics_slot) -> None:
    """Live render + telemetry. Reruns on its own timer (see
    RENDER_TICK_SECONDS) without rerunning the rest of the page — this is
    what keeps auto-play from flickering the whole app and resetting scroll
    position on every step. render_slot/metrics_slot are `st.container()`s
    created once in main() (one next to the render pane, one next to the
    controls) so this single fragment updates both locations on every tick
    instead of stacking metrics below a full-width image.
    """
    ss = st.session_state

    if ss.running and not ss.done:
        now = time.monotonic()
        if now - ss.last_step_time >= 1.0 / fps:
            take_step(condition, attack_level)
            ss.last_step_time = now

    with render_slot:
        if ss.worker_error:
            st.error(f"Render subprocess crashed: {ss.worker_error}")
        else:
            # Fixed display width, well under the source 960x960 render --
            # shrinks the on-page footprint without touching render quality
            # (the browser downsamples from the full-resolution frame).
            st.image(ss.last_frame, channels="rgb", width=560)
            st.write(f"Step {ss.step_count} / {config.MAX_EPISODE_STEPS_PICKANDPLACE}")
            if ss.done:
                if ss.success:
                    st.success("Episode SUCCESS")
                else:
                    st.error("Episode FAILURE")

    if ss.worker_error:
        return

    with metrics_slot:
        st.subheader("Live metrics")
        st.metric("Distance to goal", f"{ss.distance_to_goal_now:.3f} m")
        jcol1, jcol2 = st.columns(2)
        jcol1.metric("Arm jerk", f"{ss.arm_jerk:.2f}")
        jcol2.metric("Grip jerk", f"{ss.grip_jerk:.2f}")
        if ss.safety_violation_step:
            st.warning("C4 safety violation this step")
        else:
            st.caption("No C4 safety violation this step")

        st.divider()
        st.subheader("Classifier / Recovery v4")
        if ss.class_probs:
            predicted_label = max(ss.class_probs, key=ss.class_probs.get)
            st.write(f"Predicted: **{predicted_label}** (p={ss.class_probs[predicted_label]:.2f})")
            st.progress(min(max(ss.p_fail, 0.0), 1.0), text=f"p_fail = {ss.p_fail:.3f}")
            st.caption(f"EMA(p_fail) = {ss.ema_pfail:.3f}")
            st.progress(
                min(max(ss.recovery_weight, 0.0), 1.0),
                text=f"Recovery v4 blend weight w = {ss.recovery_weight:.3f}",
            )
        else:
            st.caption(
                "Classifier telemetry needs 1 step of history — available "
                "from step 2 onward."
            )


def main() -> None:
    st.set_page_config(page_title="TAIRO Live Attack Demo", layout="wide")
    init_state()
    st.session_state.policy = load_policy()

    st.title("SAC+HER (clean_2M) — Live Attack Demo")

    col_render, col_controls = st.columns([1, 1])

    with col_controls:
        st.subheader("Attack controls")
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

        st.caption(
            "Changing condition/magnitude mid-episode takes effect from the "
            "next step onward (no Reset required) — a known v1 simplification, "
            "not a bug."
        )

        seed = st.number_input("Episode seed", value=0, step=1, key="seed_input")

        st.divider()
        metrics_slot = st.container()

    with col_render:
        render_slot = st.container()

        st.subheader("Playback")
        prow1 = st.columns(2)
        prow2 = st.columns(2)
        if prow1[0].button("Start", disabled=st.session_state.done, width="stretch"):
            st.session_state.running = True
        if prow1[1].button("Pause", width="stretch"):
            st.session_state.running = False
        if prow2[0].button("Step", disabled=st.session_state.done, width="stretch"):
            take_step(condition, attack_level)
        if prow2[1].button("Reset", width="stretch"):
            reset_episode(int(seed))

    render_fragment(condition, attack_level, PLAYBACK_FPS, render_slot, metrics_slot)


if __name__ == "__main__":
    main()
