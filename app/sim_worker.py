"""
Subprocess worker that owns the MuJoCo FetchPickAndPlace-v4 env and does all
env.step()/env.render() calls.

Why this exists: MuJoCo's offscreen rgb_array rendering on macOS only works
via the GLFW backend, and GLFW's OpenGL context creation is main-thread-only
on macOS (a Cocoa/AppKit restriction) -- creating it from any other thread
crashes the process with SIGTRAP, not a catchable Python exception. Streamlit
always executes the app script body in a background ScriptRunner thread
(streamlit/runtime/scriptrunner/script_runner.py), never the process's real
main thread, so rendering directly inside the Streamlit script crashes.
Running the env in its own subprocess sidesteps this: a subprocess gets its
own genuine main thread, so GLFW context creation there is safe. Confirmed
this crash reproduces with a bare threading.Thread (no Streamlit involved)
and does not reproduce when the same code runs in a separate process.

Only env creation/stepping/rendering lives here. Policy inference (SB3
`model.predict`) has no GL/thread affinity and stays in the Streamlit
process -- only actions cross the pipe, not observations needing policy
calls.

The worker logs its own progress to WORKER_LOG_PATH (opened/flushed per
write, not relying on stdout inheritance) because a hard native crash
(e.g. a GL context fault) kills the process without unwinding through
Python's exception machinery -- ordinary try/except in the parent only
ever sees a bare EOFError on the pipe. The log lets us see how far the
child got before dying even when the crash itself isn't catchable.
"""

import multiprocessing as mp
import os
import pickle
import threading
import time
import traceback

import mujoco
import numpy as np

import config

WORKER_LOG_PATH = "/tmp/tairo_sim_worker.log"


def _log_path(name: str) -> str:
    """Per-worker log path so two simultaneous workers (recovery-comparison
    mode) don't interleave indistinguishably in one shared file. Empty name
    (the default, single-worker case) preserves the original path exactly.
    """
    if not name:
        return WORKER_LOG_PATH
    root, ext = os.path.splitext(WORKER_LOG_PATH)
    return f"{root}_{name}{ext}"

# Demo-only render tuning (cosmetic; does not affect obs/physics used by the
# policy or attacks). The gymnasium-robotics Fetch envs default to a 480x480
# frame from a distant, wide-angle camera -- fine for batch video recording,
# but blocky and hard to see when stretched across a wide demo pane. Bumped
# resolution + a closer, slightly lower camera reads much more clearly.
# Chosen by visually comparing rendered frames at a few (distance, elevation)
# settings; anything closer than distance=1.6 starts clipping the arm.
DEMO_RENDER_SIZE = 960
DEMO_CAMERA_CONFIG = {
    "distance": 1.6,
    "azimuth": 132.0,
    "elevation": -18.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


def _log(msg: str, log_path: str = WORKER_LOG_PATH) -> None:
    with open(log_path, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [pid {os.getpid()}] {msg}\n")
        f.flush()


def _make_render_env(seed: int):
    import gymnasium as gym

    env = gym.make(
        config.ENV_ID_PICKANDPLACE,
        max_episode_steps=config.MAX_EPISODE_STEPS_PICKANDPLACE,
        render_mode="rgb_array",
        width=DEMO_RENDER_SIZE,
        height=DEMO_RENDER_SIZE,
        default_camera_config=DEMO_CAMERA_CONFIG,
    )
    # Do NOT call env.reset() here -- caller resets with a specific seed.
    return env


def _render_preserving_physics(env, scratch_data):
    """Render a frame without letting rendering perturb the physics state
    the next env.step() sees.

    Root cause (found via direct A/B, then confirmed with a full MjData
    snapshot/restore): gymnasium_robotics's MujocoFetchEnv._render_callback
    -- called by every env.render() -- repositions the purely-cosmetic
    goal-marker site and then calls mujoco.mj_forward() to apply it. But
    mj_forward() also re-runs MuJoCo's contact/constraint solver, which
    measurably changes data.qacc_warmstart and other solver-internal
    fields as a side effect -- confirmed on a real seed/condition where
    this flips an episode from FAIL to SUCCESS purely because the demo
    renders every step and the headless batch eval (evaluation/
    episode_runner.py) does not. Restoring data.qacc_warmstart alone was
    NOT sufficient (still perturbed the outcome); only a full MjData
    snapshot/restore around render() reproduces the same trajectory as
    never rendering at all. This is upstream gymnasium_robotics behavior
    (mj_forward is not side-effect-free between mj_step calls, contrary
    to what its name suggests), not something fixable there -- worked
    around here by treating render() as the side-effect-free operation
    it's assumed to be, via a full physics-state snapshot around it.

    scratch_data is a pre-allocated MjData buffer for the current env's
    model, reused across calls to avoid a fresh allocation every frame.
    """
    model = env.unwrapped.model
    data = env.unwrapped.data
    mujoco.mj_copyData(scratch_data, model, data)
    frame = env.render()
    mujoco.mj_copyData(data, model, scratch_data)
    return frame


def _worker_loop(conn, seed: int, log_path: str = WORKER_LOG_PATH) -> None:
    _log("worker starting", log_path)
    try:
        env = _make_render_env(seed)
        scratch_data = mujoco.MjData(env.unwrapped.model)
        _log("env created", log_path)
        obs, _info = env.reset(seed=seed)
        _log("env reset ok", log_path)
        frame = _render_preserving_physics(env, scratch_data)
        _log(f"first render ok, frame shape={frame.shape}", log_path)
        conn.send(("ready", obs, frame))

        while True:
            msg = conn.recv()
            cmd = msg[0]

            if cmd == "step":
                action = msg[1]
                obs, reward, terminated, truncated, info = env.step(action)
                frame = _render_preserving_physics(env, scratch_data)
                conn.send(("stepped", obs, reward, terminated, truncated, info, frame))

            elif cmd == "reset":
                seed = msg[1]
                env.close()
                env = _make_render_env(seed)
                scratch_data = mujoco.MjData(env.unwrapped.model)
                obs, _info = env.reset(seed=seed)
                frame = _render_preserving_physics(env, scratch_data)
                conn.send(("ready", obs, frame))

            elif cmd == "close":
                env.close()
                conn.close()
                break
    except Exception:
        tb = traceback.format_exc()
        _log(f"CRASHED:\n{tb}", log_path)
        try:
            conn.send(("error", tb))
        except Exception:
            pass
        raise


class SimWorkerCrashed(RuntimeError):
    """Raised when the render subprocess dies (crash or otherwise)."""


# multiprocessing's spawn-based Process.start() reduces the child's Pipe
# handle via a process-GLOBAL "currently spawning Popen" slot (see
# multiprocessing/context.py's set_spawning_popen/get_spawning_popen,
# used internally by popen_spawn_posix during argument pickling). Streamlit
# runs each browser session's script in its own thread of the SAME process,
# so two sessions landing at nearly the same instant (e.g. two tabs opened
# together) can call SimWorker.__init__ concurrently from different
# threads -- racing on that global slot and corrupting which pipe file
# descriptor gets wired into which child. Confirmed via a real multi-session
# stress test: concurrent session startup produced _pickle.UnpicklingError
# ("invalid load key", "could not find MARK") on the parent's recv() and
# BrokenPipeError on the child's send(), consistently reproducing whenever
# multiple SimWorkers were constructed at the same time. Tried narrowing
# the lock to just Pipe()+Process.start() (releasing before the ~1-2s
# handshake recv()) to let concurrent cold-starts overlap more -- this
# measurably made things WORSE (0/4 sessions completed vs. this wider
# scope's 4/4 and 4/4 on repeat runs), so the lock deliberately holds
# through the initial handshake too, fully serializing each session's
# cold start. Once running, each session's step()/reset() calls only
# touch its own already-established pipe and stay fully concurrent --
# only the ~1-2s-per-worker startup is serialized, not gameplay.
_SPAWN_LOCK = threading.Lock()


class SimWorker:
    """Owns a subprocess running the MuJoCo env; talks to it over a Pipe."""

    def __init__(self, seed: int = 0, name: str = "") -> None:
        self._log_path = _log_path(name)
        ctx = mp.get_context("spawn")
        with _SPAWN_LOCK:
            self._parent_conn, child_conn = ctx.Pipe()
            self._process = ctx.Process(
                target=_worker_loop, args=(child_conn, seed, self._log_path), daemon=True
            )
            self._process.start()
            _tag, self.obs, self.frame = self._recv()

    def _recv(self):
        try:
            msg = self._parent_conn.recv()
        except (EOFError, OSError, pickle.PickleError) as e:
            # EOFError = pipe closed (process gone). OSError/PickleError =
            # the pipe delivered a truncated or garbled byte stream --
            # confirmed via the same stress test that surfaced the
            # _SPAWN_LOCK race above: before that fix, a corrupted read
            # here raised _pickle.UnpicklingError uncaught, crashing the
            # whole Streamlit session with a raw traceback instead of this
            # class's intended graceful error. The lock removes the known
            # cause; this still catches the failure mode defensively.
            raise SimWorkerCrashed(
                f"Render subprocess died unexpectedly (exitcode={self._process.exitcode}). "
                f"See {self._log_path} for details."
            ) from e
        if msg[0] == "error":
            raise SimWorkerCrashed(f"Render subprocess raised an exception:\n{msg[1]}")
        return msg

    def reset(self, seed: int):
        self._parent_conn.send(("reset", seed))
        _tag, self.obs, self.frame = self._recv()
        return self.obs, self.frame

    def step(self, action):
        self._parent_conn.send(("step", action))
        _tag, obs, reward, terminated, truncated, info, frame = self._recv()
        self.obs = obs
        self.frame = frame
        return obs, reward, terminated, truncated, info, frame

    def close(self) -> None:
        try:
            self._parent_conn.send(("close",))
        except Exception:
            pass
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
