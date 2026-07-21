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
import time
import traceback

import numpy as np

import config

WORKER_LOG_PATH = "/tmp/tairo_sim_worker.log"

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


def _log(msg: str) -> None:
    with open(WORKER_LOG_PATH, "a") as f:
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


def _worker_loop(conn, seed: int) -> None:
    _log("worker starting")
    try:
        env = _make_render_env(seed)
        _log("env created")
        obs, _info = env.reset(seed=seed)
        _log("env reset ok")
        frame = env.render()
        _log(f"first render ok, frame shape={frame.shape}")
        conn.send(("ready", obs, frame))

        while True:
            msg = conn.recv()
            cmd = msg[0]

            if cmd == "step":
                action = msg[1]
                obs, reward, terminated, truncated, info = env.step(action)
                frame = env.render()
                conn.send(("stepped", obs, reward, terminated, truncated, info, frame))

            elif cmd == "reset":
                seed = msg[1]
                env.close()
                env = _make_render_env(seed)
                obs, _info = env.reset(seed=seed)
                frame = env.render()
                conn.send(("ready", obs, frame))

            elif cmd == "close":
                env.close()
                conn.close()
                break
    except Exception:
        tb = traceback.format_exc()
        _log(f"CRASHED:\n{tb}")
        try:
            conn.send(("error", tb))
        except Exception:
            pass
        raise


class SimWorkerCrashed(RuntimeError):
    """Raised when the render subprocess dies (crash or otherwise)."""


class SimWorker:
    """Owns a subprocess running the MuJoCo env; talks to it over a Pipe."""

    def __init__(self, seed: int = 0) -> None:
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._process = ctx.Process(
            target=_worker_loop, args=(child_conn, seed), daemon=True
        )
        self._process.start()
        _tag, self.obs, self.frame = self._recv()

    def _recv(self):
        try:
            msg = self._parent_conn.recv()
        except EOFError as e:
            raise SimWorkerCrashed(
                f"Render subprocess died unexpectedly (exitcode={self._process.exitcode}). "
                f"See {WORKER_LOG_PATH} for details."
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
