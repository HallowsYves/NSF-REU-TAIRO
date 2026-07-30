"""
Record episodes of FetchPickAndPlace-v4 for each (policy, condition) combination,
spread evenly across the canonical sweep seeds (config.RANDOM_SEEDS).

Videos are saved as .mp4 files under:
  results/videos/<condition>/<policy_name>/
  Named: {policy}_{condition}_seed{N}-episode-{M}.mp4

Delegates episode execution to evaluation.episode_runner.run_episode() (the
same function the canonical sweep uses) so recorded behavior is guaranteed
consistent with sweep results, rather than a second, hand-rolled episode
loop. The env is wrapped with gymnasium's RecordVideo before being passed
to run_episode(), which drives it with ordinary env.reset()/env.step()
calls.

Seeding note
------------
In the full sweep, every episode within a seed block resets with the same
seed value (env.reset(seed=seed)), making all 30 repetitions per seed
bit-identical.  The 5 seeds (RANDOM_SEEDS) are where scenario diversity
comes from — each represents a different goal position.  Recording here
therefore samples eps_per_seed episodes per seed; they are identical to
each other within a seed but show different scenarios across seeds.

--n-episodes is the TOTAL episodes per (policy, condition) pair, divided
evenly across seeds.  Default 10 → 2 episodes per seed × 5 seeds.
If --n-episodes is not divisible by the number of seeds, the actual total
is rounded down to eps_per_seed * len(RANDOM_SEEDS).

Recovery v4 / HX6
------------------
"sac_her_recovery_v4_hx6" is supported alongside sac_her/v2/v3. It requires
the same classifier artifacts as run_multiseed_sweep.py and is scoped to the
clean_2M checkpoint only (see RECOVERY_V4.md) — do not point --model-path
at a different checkpoint when recording hx6.

Usage:
  conda run -n reu_robotics python3 scripts/record_videos.py
  conda run -n reu_robotics python3 scripts/record_videos.py \\
      --conditions sensor_dropout action_reversal \\
      --policies sac_her sac_her_recovery_v3 \\
      --n-episodes 10
  conda run -n reu_robotics python3 scripts/record_videos.py \\
      --conditions action_clipping \\
      --policies sac_her sac_her_recovery_v4_hx6 \\
      --n-episodes 1 \\
      --recovery-v4-classifier-dir results/classifier_seedfix
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import SAC

from config import (
    ALL_CONDITIONS, ATTACK_LEVELS, CLASSIFIER_DIR, MAX_EPISODE_STEPS,
    RANDOM_SEEDS, RESULTS_DIR,
)
from envs.fetchpickandplace_env import make_env
from evaluation.episode_runner import run_episode

# ---------------------------------------------------------------------------
# Canonical lists
# ---------------------------------------------------------------------------

ALL_POLICIES = [
    "sac_her",
    "sac_her_recovery_v2",
    "sac_her_recovery_v3",
    "sac_her_recovery_v4_hx6",
]

# Recovery policies skip the clean condition (consistent with the sweep)
_RECOVERY_POLICIES = {"sac_her_recovery_v2", "sac_her_recovery_v3", "sac_her_recovery_v4_hx6"}

_LEVEL4_CLASSIFIER_PATH = os.path.join(RESULTS_DIR, "classifier_level4", "level4_classifier.pkl")


# ---------------------------------------------------------------------------
# Single (policy, condition) block
# ---------------------------------------------------------------------------

def record_pair(
    policy_name, condition, model, n_episodes, output_dir,
    recovery_v4_classifier=None, recovery_v4_calibration=None,
    recovery_v4_checkpoint="clean_2M", level4_classifier=None,
    seeds=RANDOM_SEEDS, episode_offset=0,
):
    """Record episodes for one (policy_name, condition), spread across RANDOM_SEEDS.

    n_episodes is the total target; it is divided by len(RANDOM_SEEDS) to get
    eps_per_seed.  A separate RecordVideo env is created per seed so that video
    filenames embed the seed: {policy}_{condition}_seed{N}-episode-{M}.mp4.

    All episodes within a seed are identical (the sweep resets with the same
    seed each time), so eps_per_seed > 1 shows the same scenario repeatedly —
    useful for confirming consistency but not for adding scenario variety.
    """
    video_folder = os.path.join(output_dir, condition, policy_name)
    os.makedirs(video_folder, exist_ok=True)

    attack_level = ATTACK_LEVELS[condition]

    # Divide total episodes evenly across seeds; round down if not divisible.
    eps_per_seed = max(1, n_episodes // len(seeds))

    successes = []

    for seed in seeds:
        # Fresh env + wrapper per seed so filenames embed the seed number.
        base_env = make_env(seed=seed, rgb_mode=True)
        env = RecordEpisodeStatistics(base_env, buffer_length=eps_per_seed)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=f"{policy_name}_{condition}_seed{seed}",
            episode_trigger=lambda ep: True,
        )

        try:
            for within_seed_ep in range(eps_per_seed):
                episode_in_seed = within_seed_ep + episode_offset
                result, _step_df = run_episode(
                    env=env,
                    method=policy_name,
                    seed=seed,
                    episode_in_seed=episode_in_seed,
                    condition=condition,
                    attack_level=attack_level,
                    model=model,
                    max_steps=MAX_EPISODE_STEPS,
                    recovery_v4_classifier=recovery_v4_classifier,
                    recovery_v4_calibration=recovery_v4_calibration,
                    recovery_v4_checkpoint=recovery_v4_checkpoint,
                    level4_classifier=level4_classifier,
                )
                successes.append(bool(result.success))
                print(
                    f"  seed={seed} episode_in_seed={episode_in_seed} | "
                    f"success={bool(result.success)} | reward={result.total_reward:.2f}"
                )
        finally:
            env.close()

    rate = sum(successes) / len(successes) if successes else 0.0
    print(f"  Success rate: {rate:.1%} ({sum(successes)}/{len(successes)})\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record FetchPickAndPlace-v4 episode videos for each (policy, condition)."
    )
    parser.add_argument(
        "--conditions", nargs="+", default=ALL_CONDITIONS,
        metavar="CONDITION",
        help="Conditions to record. Default: all.",
    )
    parser.add_argument(
        "--policies", nargs="+", default=ALL_POLICIES,
        metavar="POLICY",
        help="Policies to record. Default: all.",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help=(
            "Total episodes per (policy, condition) pair, spread across seeds. "
            "Divided by len(RANDOM_SEEDS) to get episodes-per-seed; rounded down "
            "if not evenly divisible. Default 10 → 2 per seed × 5 seeds."
        ),
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(RESULTS_DIR, "videos"),
        help="Root output directory. Videos go in <output-dir>/<condition>/<policy>/",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=RANDOM_SEEDS,
        metavar="SEED",
        help="Seeds to record from (subset of RANDOM_SEEDS). Default: all 5. "
             "Pass a single seed (e.g. --seeds 0) to produce exactly one video "
             "per (policy, condition) instead of one per seed.",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(RESULTS_DIR, "models", "sac_her_pickandplace_clean_2M"),
        help="Path to the SAC+HER model zip (without .zip extension).",
    )
    parser.add_argument(
        "--episode-offset", type=int, default=0,
        help="Added to each within-seed episode index before calling "
             "run_episode(episode_in_seed=...). Use this to target a specific "
             "episode (e.g. one known to fail in results/data_seedfix/) "
             "instead of always starting at episode 0 within a seed.",
    )
    parser.add_argument(
        "--recovery-v4-classifier-dir", default=None,
        help=(
            "Directory containing online_failure_classifier.pkl and "
            "recovery_v4_trigger_calibration.pkl. Required for "
            "sac_her_recovery_v4_hx6. Defaults to CLASSIFIER_DIR from "
            "config.py, which does NOT contain the seed-fixed artifacts — "
            "pass 'results/classifier_seedfix' explicitly (mirrors "
            "run_multiseed_sweep.py)."
        ),
    )
    parser.add_argument(
        "--recovery-v4-calibration-dir", default=None,
        help="Defaults to --recovery-v4-classifier-dir if not given.",
    )
    parser.add_argument(
        "--recovery-v4-checkpoint", default="clean_2M",
        help="Key into recovery_v4_trigger_calibration.pkl. Default clean_2M "
             "(HX6's only supported scope).",
    )
    args = parser.parse_args()

    # Load model once; reuse across all (policy, condition) pairs
    load_env = make_env(seed=0)
    model = SAC.load(args.model_path, env=load_env)
    load_env.close()

    # Load Recovery v4 / HX6 artifacts, once, only if requested.
    recovery_v4_classifier = None
    recovery_v4_calibration = None
    level4_classifier = None
    if "sac_her_recovery_v4_hx6" in args.policies:
        v4_classifier_dir = args.recovery_v4_classifier_dir or CLASSIFIER_DIR
        v4_calibration_dir = args.recovery_v4_calibration_dir or v4_classifier_dir
        classifier_path = os.path.join(v4_classifier_dir, "online_failure_classifier.pkl")
        calibration_path = os.path.join(v4_calibration_dir, "recovery_v4_trigger_calibration.pkl")
        print("Loading Recovery v4 / HX6 artifacts:")
        print(f"  {classifier_path}")
        print(f"  {calibration_path}")
        print(f"  {_LEVEL4_CLASSIFIER_PATH}")
        with open(classifier_path, "rb") as f:
            recovery_v4_classifier = pickle.load(f)
        with open(calibration_path, "rb") as f:
            recovery_v4_calibration = pickle.load(f)
        if args.recovery_v4_checkpoint not in recovery_v4_calibration:
            raise ValueError(
                f"--recovery-v4-checkpoint='{args.recovery_v4_checkpoint}' has no "
                f"entry in {calibration_path} "
                f"(available: {list(recovery_v4_calibration.keys())})"
            )
        with open(_LEVEL4_CLASSIFIER_PATH, "rb") as f:
            level4_classifier = pickle.load(f)

    for policy_name in args.policies:
        for condition in args.conditions:
            if condition == "clean" and policy_name in _RECOVERY_POLICIES:
                print(f"Skipping clean for {policy_name} (consistent with sweep).")
                continue

            print(f"\n=== {policy_name} | {condition} ===")
            record_pair(
                policy_name=policy_name,
                condition=condition,
                model=model,
                n_episodes=args.n_episodes,
                output_dir=args.output_dir,
                recovery_v4_classifier=recovery_v4_classifier,
                recovery_v4_calibration=recovery_v4_calibration,
                recovery_v4_checkpoint=args.recovery_v4_checkpoint,
                level4_classifier=level4_classifier,
                seeds=args.seeds,
                episode_offset=args.episode_offset,
            )

    print("Done. Videos saved to:", args.output_dir)


if __name__ == "__main__":
    main()
