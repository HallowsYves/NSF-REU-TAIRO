"""
Live interactive demo: SAC+HER (clean_2M) on FetchPickAndPlace-v4 under
adversarial attack conditions, rendered step-by-step in a Streamlit page.

Run with:
    conda activate reu_robotics
    streamlit run app/live_attack_demo.py

v1 scope: live render + attack controls (condition + magnitude/toggle) +
step counter + end-of-episode success/fail readout only.

v2 (deferred, not implemented here): a metrics panel (distance-to-goal via
envs.fetchpickandplace_env.distance_to_goal, C4 jerk/safety scoring) and a
classifier/recovery overlay (online failure-mode classifier's live p_fail,
Recovery v4 trigger state).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
from stable_baselines3 import SAC

import config
from envs.fetchpickandplace_env import make_env
from policies.sac_her_policy import SACHerPolicy
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack
from app.sim_worker import SimWorker, SimWorkerCrashed

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


@st.cache_resource
def load_policy() -> SACHerPolicy:
    load_env = make_env(seed=0)
    model = SAC.load(config.MODEL_PATH_PICKANDPLACE_2M, env=load_env)
    load_env.close()
    return SACHerPolicy(model)


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

    try:
        ss.obs, _reward, terminated, truncated, info, ss.last_frame = ss.worker.step(
            executed_action
        )
    except SimWorkerCrashed as e:
        ss.worker = None
        ss.worker_error = str(e)
        ss.done = True
        ss.running = False
        return

    ss.step_count += 1

    if terminated or truncated:
        ss.done = True
        ss.running = False
        ss.success = bool(info.get("is_success", False))


@st.fragment(run_every=RENDER_TICK_SECONDS)
def render_fragment(condition: str, attack_level: float, fps: int) -> None:
    """Live render pane. Reruns on its own timer (see RENDER_TICK_SECONDS)
    without rerunning the rest of the page — this is what keeps auto-play
    from flickering the whole app and resetting scroll position on every
    step; only this fragment's own container redraws each tick.
    """
    ss = st.session_state

    if ss.running and not ss.done:
        now = time.monotonic()
        if now - ss.last_step_time >= 1.0 / fps:
            take_step(condition, attack_level)
            ss.last_step_time = now

    if ss.worker_error:
        st.error(f"Render subprocess crashed: {ss.worker_error}")
        return

    st.image(ss.last_frame, channels="rgb", width="stretch")
    st.write(f"Step {ss.step_count} / {config.MAX_EPISODE_STEPS_PICKANDPLACE}")

    if ss.done:
        if ss.success:
            st.success("Episode SUCCESS")
        else:
            st.error("Episode FAILURE")


def main() -> None:
    st.set_page_config(page_title="TAIRO Live Attack Demo", layout="wide")
    init_state()
    st.session_state.policy = load_policy()

    st.title("SAC+HER (clean_2M) — Live Attack Demo")

    col_render, col_controls = st.columns([2, 1])

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
        fps = st.slider("Target FPS", min_value=1, max_value=4, value=4, key="fps")

        st.subheader("Playback")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Start", disabled=st.session_state.done):
            st.session_state.running = True
        if b2.button("Pause"):
            st.session_state.running = False
        if b3.button("Step", disabled=st.session_state.done):
            take_step(condition, attack_level)
        if b4.button("Reset"):
            reset_episode(int(seed))

    with col_render:
        render_fragment(condition, attack_level, fps)


if __name__ == "__main__":
    main()
