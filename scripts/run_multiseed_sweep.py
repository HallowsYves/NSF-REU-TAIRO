"""
TAIRO B0–B3 multi-seed benchmark sweep.

Iterates over seeds × conditions × methods × recovery versions and writes
one row per episode (episode_results.csv) and one row per timestep
(step_logs.csv) to results/data/.

Benchmark layer assignment
--------------------------
B0 : condition == "clean", no recovery        (clean SAC+HER baseline)
B1 : all conditions,       no recovery        (attacked, base policies)
B2 : all conditions,       recovery_v2
B3 : all conditions,       recovery_v3        (current best)

Output filenames
----------------
FetchReach (default):
    results/data/episode_results.csv          ← same as before (backward-compat)
    results/data/step_logs.csv

PickAndPlace:
    results/data/episode_results_pickandplace_clean.csv
    results/data/step_logs_pickandplace_clean.csv
    results/data/episode_results_pickandplace_randomized.csv   (--randomized)
    results/data/step_logs_pickandplace_randomized.csv

Usage examples
--------------
FetchReach (default — behaviour unchanged):
    conda run -n reu_robotics python3 scripts/run_multiseed_sweep.py

PickAndPlace, default clean model:
    conda run -n reu_robotics python3 scripts/run_multiseed_sweep.py --env pickandplace

PickAndPlace, specific model, 1 seed, 2 episodes (verification):
    conda run -n reu_robotics python3 scripts/run_multiseed_sweep.py \\
        --env pickandplace \\
        --model-path results/models/sac_her_pickandplace_clean_500k \\
        --seeds 0 \\
        --n-episodes 2

PickAndPlace, domain-randomized model:
    conda run -n reu_robotics python3 scripts/run_multiseed_sweep.py \\
        --env pickandplace --randomized
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import (
    RANDOM_SEEDS,
    ALL_CONDITIONS,
    DEFAULT_METHODS,
    ATTACK_LEVELS,
    N_EPISODES_PER_SEED,
    DATA_DIR,
    MAX_EPISODE_STEPS,
    MAX_EPISODE_STEPS_PICKANDPLACE,
    MODEL_PATH,
    MODEL_PATH_PICKANDPLACE,
    MODEL_PATH_PICKANDPLACE_RANDOMIZED,
    SB3_AVAILABLE,
)
from evaluation.episode_runner import run_episode

_RECOVERY_VERSION = {
    "sac_her":                 "none",
    "sac_her_recovery_v2":     "v2",
    "sac_her_recovery_v3":     "v3",
    "sac_her_recovery_v4":     "v4",
    "sac_her_recovery_v4_hx":  "v4_hx",
    "sac_her_recovery_v4_hx2": "v4_hx2",
    "sac_her_recovery_v4_hx3": "v4_hx3",
}


def _layer(method: str, condition: str) -> str:
    if method == "sac_her":
        return "B0" if condition == "clean" else "B1"
    if method == "sac_her_recovery_v2":
        return "B2"
    if method in {"sac_her_recovery_v4", "sac_her_recovery_v4_hx",
                  "sac_her_recovery_v4_hx2", "sac_her_recovery_v4_hx3"}:
        return "B4"  # v4_hx/v4_hx2/v4_hx3 are same-layer variants of v4, not new benchmark tiers
    return "B3"  # sac_her_recovery_v3


def _parse_args():
    parser = argparse.ArgumentParser(
        description="TAIRO B0–B3 multi-seed benchmark sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        choices=["fetchreach", "pickandplace"],
        default="fetchreach",
        help="Environment to sweep (default: fetchreach).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Override the model file to load (no .zip extension). "
            "If omitted, uses MODEL_PATH for fetchreach or "
            "MODEL_PATH_PICKANDPLACE for pickandplace. "
            "Example: --model-path results/models/sac_her_pickandplace_clean_500k"
        ),
    )
    parser.add_argument(
        "--randomized",
        action="store_true",
        help=(
            "Use MODEL_PATH_PICKANDPLACE_RANDOMIZED as default model. "
            "Only meaningful with --env pickandplace. "
            "Ignored if --model-path is also set."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help=(
            "Seeds to evaluate (space-separated). "
            "Default: all RANDOM_SEEDS from config.py. "
            "Example: --seeds 0 1 2"
        ),
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        metavar="CONDITION",
        choices=ALL_CONDITIONS,
        help=(
            "Conditions to evaluate (space-separated). Default: all "
            "ALL_CONDITIONS from config.py. Added for targeted statistical- "
            "power re-runs (e.g. re-running only 2-3 conditions with more "
            "seeds) without paying the full 11-condition grid's runtime. "
            "Example: --conditions grip_state_falsification action_delay"
        ),
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Episodes per (seed × condition × method). "
            "Default: N_EPISODES_PER_SEED from config.py. "
            "Example: --n-episodes 2"
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        metavar="METHOD",
        choices=["sac_her", "sac_her_recovery_v2", "sac_her_recovery_v3",
                 "sac_her_recovery_v4", "sac_her_recovery_v4_hx", "sac_her_recovery_v4_hx2",
                 "sac_her_recovery_v4_hx3"],
        help=(
            "Methods to evaluate (space-separated). "
            "Default: DEFAULT_METHODS from config.py (sac_her, "
            "recovery_v2, recovery_v3 -- excludes all sac_her_recovery_v4* "
            "variants so the bare command stays reproducible). "
            "Pass --methods sac_her_recovery_v4 (or _hx / _hx2 / _hx3) explicitly to "
            "opt in; all four require results/classifier_seedfix/online_failure_classifier.pkl "
            "and recovery_v4_trigger_calibration.pkl to exist and are only "
            "calibrated for the clean_2M PickAndPlace checkpoint "
            "(--recovery-v4-checkpoint). sac_her_recovery_v4_hx additionally "
            "gates recovery_v4's expert mixture by an online Level-1 task-stage "
            "signal (see recovery/recovery_v4_hx.py); sac_her_recovery_v4_hx2 "
            "layers a Level-4 attack-family down-weight on top of that and "
            "additionally requires results/classifier_level4/level4_classifier.pkl "
            "(see recovery/recovery_v4_hx2.py); sac_her_recovery_v4_hx3 additionally "
            "re-gates relocalization_expert on Level 4's perception_state signal "
            "(same level4_classifier.pkl requirement as hx2, see recovery/recovery_v4_hx3.py). "
            "Example: --methods sac_her"
        ),
    )
    parser.add_argument(
        "--recovery-v4-checkpoint",
        type=str,
        default="clean_2M",
        metavar="CHECKPOINT",
        help=(
            "Which trained-model checkpoint name to look up in "
            "recovery_v4_trigger_calibration.pkl when method="
            "sac_her_recovery_v4 is being run. Only meaningful if that "
            "method is included. Tier 1 CCAR is currently scoped to "
            "clean_2M only -- see RECOVERY_V4.md section 2.6. Must match "
            "whichever model --model-path actually points at; this is not "
            "auto-derived from --model-path."
        ),
    )
    parser.add_argument(
        "--recovery-v4-classifier-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory holding online_failure_classifier.pkl (and, unless "
            "--recovery-v4-calibration-dir is given, recovery_v4_trigger_"
            "calibration.pkl) for sac_her_recovery_v4. "
            "Default: CLASSIFIER_DIR from config.py (results/classifier/). "
            "The seed-fixed artifacts live in results/classifier_seedfix/, "
            "so pass that here when running v4 on the seed-fixed benchmark. "
            "For the Phase C dense-classifier run, pass "
            "results/classifier_seedfix_dense/ here."
        ),
    )
    parser.add_argument(
        "--recovery-v4-calibration-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory holding recovery_v4_trigger_calibration.pkl, decoupled "
            "from --recovery-v4-classifier-dir so the trigger calibration "
            "(midpoint/alpha/K) can be held FIXED while the online classifier "
            "is swapped. Default: falls back to --recovery-v4-classifier-dir "
            "(original coupled behavior). Phase C uses this to run the dense "
            "classifier against the unchanged classifier_seedfix calibration."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory to write episode_results/step_logs CSVs to. "
            "Default: DATA_DIR from config.py (results/data/). "
            "Use a distinct directory (e.g. results/data_seedfix/) to avoid "
            "overwriting existing CSVs from a prior run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Resolve environment, model path, step budget, and output tag.
    # FetchReach path is unchanged from the pre-flag version.
    # ------------------------------------------------------------------
    if args.env == "pickandplace":
        from envs.fetchpickandplace_env import make_env

        # --model-path takes precedence; --randomized selects the default.
        if args.model_path is not None:
            model_path = args.model_path
            # Derive a readable tag from the basename for the output filename.
            env_tag = os.path.splitext(os.path.basename(model_path))[0]
        elif args.randomized:
            model_path = MODEL_PATH_PICKANDPLACE_RANDOMIZED
            env_tag = "pickandplace_randomized"
        else:
            model_path = MODEL_PATH_PICKANDPLACE
            env_tag = "pickandplace_clean"

        max_steps = MAX_EPISODE_STEPS_PICKANDPLACE
        # PickAndPlace results go to separate files so FetchReach CSVs are never overwritten.
        ep_filename   = f"episode_results_{env_tag}.csv"
        step_filename = f"step_logs_{env_tag}.csv"

    else:  # fetchreach — fully backward-compatible
        from envs.fetchreach_env import make_env

        model_path = args.model_path if args.model_path is not None else MODEL_PATH
        max_steps  = MAX_EPISODE_STEPS
        env_tag    = "fetchreach"
        # Keep original filenames so existing analysis scripts are unaffected.
        ep_filename   = "episode_results.csv"
        step_filename = "step_logs.csv"

    seeds       = args.seeds      if args.seeds      is not None else RANDOM_SEEDS
    n_episodes  = args.n_episodes if args.n_episodes is not None else N_EPISODES_PER_SEED
    methods     = args.methods    if args.methods    is not None else DEFAULT_METHODS
    conditions  = args.conditions if args.conditions is not None else ALL_CONDITIONS
    output_dir  = args.output_dir if args.output_dir is not None else DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    _tmp_env = make_env(seed=0)
    model = None
    if SB3_AVAILABLE:
        from stable_baselines3 import SAC
        print(f"[sweep] env={env_tag}  max_steps={max_steps}  model={model_path}")
        print(f"[sweep] seeds={seeds}  n_episodes={n_episodes}  "
              f"conditions={len(conditions)}  methods={len(methods)}")
        model = SAC.load(model_path, env=_tmp_env)
    else:
        print("[sweep] WARNING: SB3 not available — sac_her runs will be skipped.")
    _tmp_env.close()

    # ------------------------------------------------------------------
    # Load Recovery v4 Tier 1 CCAR artifacts, once, only if requested.
    # Mirrors the model-loading pattern above -- classifier and
    # calibration are loaded once here, not per episode, and passed
    # through to every run_episode() call.
    # ------------------------------------------------------------------
    recovery_v4_classifier = None
    recovery_v4_calibration = None
    level4_classifier = None
    if any(m in methods for m in
           ("sac_her_recovery_v4", "sac_her_recovery_v4_hx",
            "sac_her_recovery_v4_hx2", "sac_her_recovery_v4_hx3")):
        import pickle
        from config import CLASSIFIER_DIR as _DEFAULT_CLASSIFIER_DIR
        v4_classifier_dir = (
            args.recovery_v4_classifier_dir
            if args.recovery_v4_classifier_dir is not None
            else _DEFAULT_CLASSIFIER_DIR
        )
        # Calibration dir is decoupled from the classifier dir so the trigger
        # calibration can be held fixed while the classifier is swapped
        # (Phase C: dense classifier + unchanged classifier_seedfix midpoint).
        v4_calibration_dir = (
            args.recovery_v4_calibration_dir
            if args.recovery_v4_calibration_dir is not None
            else v4_classifier_dir
        )
        classifier_path = os.path.join(v4_classifier_dir, "online_failure_classifier.pkl")
        calibration_path = os.path.join(v4_calibration_dir, "recovery_v4_trigger_calibration.pkl")
        print(f"[sweep] Loading Recovery v4 Tier 1 CCAR artifacts:")
        print(f"        {classifier_path}")
        print(f"        {calibration_path}")
        with open(classifier_path, "rb") as f:
            recovery_v4_classifier = pickle.load(f)
        with open(calibration_path, "rb") as f:
            recovery_v4_calibration = pickle.load(f)
        if args.recovery_v4_checkpoint not in recovery_v4_calibration:
            raise ValueError(
                f"[sweep] --recovery-v4-checkpoint='{args.recovery_v4_checkpoint}' "
                f"has no entry in {calibration_path} "
                f"(available: {list(recovery_v4_calibration.keys())})"
            )

    if "sac_her_recovery_v4_hx2" in methods or "sac_her_recovery_v4_hx3" in methods:
        level4_path = "results/classifier_level4/level4_classifier.pkl"
        print(f"[sweep] Loading Level 4 classifier artifact:\n        {level4_path}")
        with open(level4_path, "rb") as f:
            level4_classifier = pickle.load(f)

    episode_rows = []
    step_rows    = []
    episode_idx  = 0

    # ------------------------------------------------------------------
    # Main sweep loop
    # ------------------------------------------------------------------
    for seed in seeds:
        env = make_env(seed=seed)
        print(f"\n[sweep] seed={seed}")

        for condition in conditions:
            attack_level = ATTACK_LEVELS[condition]

            for method in methods:
                if model is None:
                    continue

                recovery_version = _RECOVERY_VERSION[method]
                layer = _layer(method, condition)

                for _ep in range(n_episodes):
                    result, step_df = run_episode(
                        env=env,
                        method=method,
                        condition=condition,
                        seed=seed,
                        episode_in_seed=_ep,
                        model=model,
                        attack_level=attack_level,
                        recovery_version=recovery_version,
                        max_steps=max_steps,
                        recovery_v4_classifier=recovery_v4_classifier,
                        recovery_v4_calibration=recovery_v4_calibration,
                        recovery_v4_checkpoint=args.recovery_v4_checkpoint,
                        level4_classifier=level4_classifier,
                    )

                    row = vars(result).copy()
                    row["method"]           = method
                    row["attack_level"]     = attack_level
                    row["episode_idx"]      = episode_idx
                    row["recovery_version"] = recovery_version
                    row["benchmark_layer"]  = layer
                    row["env"]              = env_tag
                    episode_rows.append(row)

                    step_df = step_df.copy()
                    step_df["episode_idx"]      = episode_idx
                    step_df["recovery_version"] = recovery_version
                    step_df["benchmark_layer"]  = layer
                    step_df["env"]              = env_tag
                    step_rows.append(step_df)

                    episode_idx += 1

        env.close()

    episode_df   = pd.DataFrame(episode_rows)
    step_df_all  = pd.concat(step_rows, ignore_index=True)

    ep_path   = os.path.join(output_dir, ep_filename)
    step_path = os.path.join(output_dir, step_filename)
    episode_df.to_csv(ep_path, index=False)
    step_df_all.to_csv(step_path, index=False)

    print(f"\n[sweep] Done — {episode_idx} episodes, {len(step_df_all)} timesteps.")
    print(f"        {ep_path}")
    print(f"        {step_path}")

    # Quick condition-coverage summary so the caller can verify the requested
    # conditions ran. Uses the first requested method as the coverage probe
    # (was hardcoded to "sac_her", which isn't always in `methods` -- e.g.
    # targeted power re-runs that only evaluate recovery variants).
    _probe_method = methods[0]
    print(f"\n[sweep] Condition coverage ({_probe_method} only):")
    probe_df = episode_df[episode_df["method"] == _probe_method]
    for cond in conditions:
        n = int((probe_df["condition"] == cond).sum())
        print(f"        {cond:<30}  {n:>3} episodes")


if __name__ == "__main__":
    main()
