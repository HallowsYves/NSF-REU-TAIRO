"""
TAIRO-HX Level 5 -- evaluate rule-based decisions against actual episode
outcomes (clean_2M only; see module docstring reasoning below).

This does NOT wire Level 5 into any runtime controller (recovery_v4.py /
episode_runner.py untouched) -- it is an offline analysis joining
results/level5_decisions.csv (this session's checkpoint-level decisions,
built from CHAINED Level 2-4 predictions) against
results/data_recovery_v4/episode_results_sac_her_pickandplace_clean_2M.csv
(actual per-method episode outcomes: sac_her / recovery_v2 / v3 / v4,
clean_2M, same 11-condition x 5-seed x 30-episode grid).

Scope: clean_2M only. Level 5's decisions elsewhere degenerate to
continue_sac_her/retry_task_stage because Level 2 has no discriminative
signal outside clean_2M (documented scope limitation, unchanged here) --
evaluating "does the decision match the outcome" off a degenerate input
signal would not be meaningful.

Episode-key join note: level5_decisions.csv's episode_idx and the v4
episode-results file's episode_idx use DIFFERENT global offsets per
condition/seed (confirmed by inspection -- both enumerate the same 30
episodes per (seed, condition) block in the same config.ALL_CONDITIONS
order, but with different starting offsets). Joining on raw episode_idx
would silently mismatch episodes. Instead, join on a LOCAL index
(episode_idx - min(episode_idx for that seed+condition)), which recovers
the true within-condition episode identity in both files.

Three-way episode partition (ground truth, from actual outcomes):
  - "no_problem":     sac_her succeeded on its own -- no recovery needed.
  - "recoverable":    sac_her failed, but recovery_v4 succeeded -- retrying/
                      compensating/reconstructing would have helped.
  - "unrecoverable":  sac_her failed AND recovery_v4 also failed -- nothing
                      in the current recovery toolkit fixes this episode.

For each partition, report what fraction of episodes ever see each Level 5
decision fire at any of their 14 checkpoints. Sanity expectations (not
enforced, just documented for reading the output):
  - "unrecoverable" should show the highest stop_safely rate.
  - "recoverable" should show the highest retry_task_stage /
    restore_trusted_goal / reconstruct_state / compensate_for_problem rate.
  - "no_problem" should show the lowest rate of any non-continue decision
    (false-alarm rate).

Input:  results/level5_decisions.csv
        results/data_recovery_v4/episode_results_sac_her_pickandplace_clean_2M.csv
Output: results/level5_outcome_eval.csv (per-episode partition + decision flags)
        printed summary tables
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

L5_PATH = "results/level5_decisions.csv"
EP_PATH = "results/data_recovery_v4/episode_results_sac_her_pickandplace_clean_2M.csv"
OUT_PATH = "results/level5_outcome_eval.csv"

NON_CONTINUE_DECISIONS = [
    "compensate_for_problem", "reconstruct_state", "retry_task_stage",
    "restore_trusted_goal", "continue_reduced_speed", "stop_safely",
]

# ── Load and restrict to clean_2M ────────────────────────────────────────
l5 = pd.read_csv(L5_PATH)
l5 = l5[l5["model"] == "clean_2M"].copy()

ep = pd.read_csv(EP_PATH)

# ── Local within-(seed,condition) episode index ──────────────────────────
# level5_decisions.csv has one method (implicitly sac_her-only rollouts,
# per hierarchical_chain_predictions.csv provenance), so local index is
# w.r.t. (seed, condition). The episode-results file additionally varies
# episode_idx's starting offset PER METHOD within the same (seed,
# condition) block (confirmed: sac_her/v2/v3/v4 each occupy their own
# consecutive 30-episode_idx range) -- localize per (method, seed,
# condition) there, else different methods' same-numbered episode gets a
# different local index and the merge silently fails (caught via the
# n_missing assert below during development).
l5["_local_idx"] = l5["episode_idx"] - l5.groupby(
    ["seed", "condition"])["episode_idx"].transform("min")
ep["_local_idx"] = ep["episode_idx"] - ep.groupby(
    ["method", "seed", "condition"])["episode_idx"].transform("min")

# ── Pivot episode outcomes to one row per episode, columns per method ───
ep_wide = ep.pivot_table(
    index=["condition", "seed", "_local_idx"],
    columns="method", values="success", aggfunc="first",
).reset_index()
ep_wide.columns.name = None
ep_wide = ep_wide.rename(columns={
    "sac_her": "success_sac_her",
    "sac_her_recovery_v2": "success_v2",
    "sac_her_recovery_v3": "success_v3",
    "sac_her_recovery_v4": "success_v4",
})

# ── Ground-truth three-way partition ─────────────────────────────────────
def partition(row):
    if row["success_sac_her"] == 1:
        return "no_problem"
    if row["success_v4"] == 1:
        return "recoverable"
    return "unrecoverable"

ep_wide["partition"] = ep_wide.apply(partition, axis=1)

# ── Per-episode Level 5 decision flags (did this decision EVER fire, at
# any of the episode's 14 checkpoints) ────────────────────────────────────
flags = l5.groupby(["condition", "seed", "_local_idx"])["level5_decision"].apply(
    lambda s: pd.Series({f"ever_{d}": int((s == d).any()) for d in NON_CONTINUE_DECISIONS})
).unstack()
flags = flags.reset_index()

merged = ep_wide.merge(flags, on=["condition", "seed", "_local_idx"], how="left")
n_missing = merged[[f"ever_{d}" for d in NON_CONTINUE_DECISIONS]].isna().any(axis=1).sum()
assert n_missing == 0, f"{n_missing} episodes failed to join to Level 5 decisions"

merged.to_csv(OUT_PATH, index=False)
print(f"[level5-eval] Saved -> {OUT_PATH}  ({len(merged)} episodes)\n")

print("=" * 70)
print("PARTITION SIZES (clean_2M, ground truth from actual episode outcomes)")
print("=" * 70)
print(merged["partition"].value_counts().to_string())
print()

print("=" * 70)
print("FRACTION OF EPISODES WHERE EACH LEVEL 5 DECISION EVER FIRES, BY PARTITION")
print("=" * 70)
rate_table = merged.groupby("partition")[[f"ever_{d}" for d in NON_CONTINUE_DECISIONS]].mean().round(3)
print(rate_table.to_string())
print()

print("=" * 70)
print("KEY CHECKS")
print("=" * 70)
for part, label in [("unrecoverable", "stop_safely"), ("recoverable", "retry_task_stage"),
                    ("no_problem", "stop_safely"), ("no_problem", "retry_task_stage")]:
    if part in rate_table.index:
        rate = rate_table.loc[part, f"ever_{label}"]
        print(f"  P(ever_{label} | partition={part}) = {rate:.3f}")

print()
print("=" * 70)
print("CROSS-TAB: partition x condition (clean_2M)")
print("=" * 70)
print(pd.crosstab(merged["condition"], merged["partition"]).to_string())

print("\n[level5-eval] Done.")
