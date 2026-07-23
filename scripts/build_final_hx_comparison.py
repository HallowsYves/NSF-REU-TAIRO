"""
Final TAIRO-HX comparison for the mentor's final-push checklist (2026-07-22,
updated 2026-07-23 for the hx6 adoption).

Compares four arms on the clean_2M PickAndPlace checkpoint, all at full
statistical power (n=450, seeds 0-14, all 11 conditions):

    no_recovery      sac_her                  -- no recovery
    v2               sac_her_recovery_v2      -- earlier recovery baseline (hard trigger)
    v3               sac_her_recovery_v3      -- earlier recovery baseline (hard trigger,
                                                  sustained window)
    gradual_v4       sac_her_recovery_v4      -- "gradual-response recovery": continuous
                                                  CCAR blend, no Level 1/4 gating (confirmed
                                                  via sign-off as the mentor's intended arm)
    selective_final  sac_her_recovery_v4_hx6  -- "final TAIRO-HX selective recovery": the
                                                  adopted variant (Level 1 stage-gate + Level 4
                                                  attack-family down-weight, same mixture as
                                                  v4_hx2, PLUS a Level-4-gated fast-attack
                                                  trigger EMA -- confirmed via sign-off to
                                                  supersede v4_hx2 after full-power evaluation
                                                  showed it matches v4_hx2 exactly everywhere
                                                  measured, including grip_state_falsification,
                                                  with zero new regression risk -- see
                                                  RECOVERY_V4.md / FINAL_APPROACH.md)

Reports the 8 mentor-requested metrics per (method, condition):
    1. task-success rate                -- episode_results['success']
    2. clean-task performance           -- (1) restricted to condition == 'clean'
    3. detection delay (steps)          -- attack onset -> first recovery_triggered==True
    4. recovery-response delay (steps)  -- first trigger -> recovery reaching full/meaningful
                                            authority. For v4-family (continuous blend): first
                                            step recovery_v4_weight >= RESPONSE_W_THRESH after
                                            the trigger. For v2/v3 (hard override): 0 by
                                            construction -- both apply a full, unblended
                                            correction on the same step they trigger, no ramp.
    5. recovery time (steps)            -- steps per episode with recovery_triggered==True
    6. safety violations                -- episode_results['safety_violation'] (C4 formula,
                                            binary per episode -- see evaluation/metrics.py)
    7. number of interventions          -- count of recovery_triggered False->True transitions
                                            per episode (distinct on/off activation events,
                                            not total duration)
    8. completion-time overhead (steps) -- mean first_success_step (successful episodes only)
                                            minus sac_her's, same condition

Metrics 1/2/6 reuse episode_results columns directly. Metric 8 reuses
episode_results['first_success_step'], already logged per episode. Metrics
3/4/5/7 are new: computed from each source's step_logs CSV (only the ~6
columns needed are read via usecols, so even the largest files here, ~600MB
on disk, stay well within memory) reduced to one row per episode, then the
raw step rows are dropped before moving to the next source.

Statistical comparison reuses bh_adjust/bootstrap_delta_ci/mcnemar_exact_pvalue/
two_proportion_ztest_pvalue from scripts/audit_recovery_do_no_harm.py --
generalized here to a single metric_col parameter (binary outcomes use
McNemar/two-proportion-z, matching the audit script's own method; continuous
delay/duration metrics use Wilcoxon signed-rank/Mann-Whitney U instead, since
McNemar only applies to binary outcomes) rather than duplicating that logic
three times over.

Output: results/final_hx_comparison_{success_safety,timing,vs_sac_her}.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from config import ALL_CONDITIONS
from evaluation.attack_dispatch import GOAL_SPOOF_MIDEP_STEP
from scripts.audit_recovery_do_no_harm import (
    bh_adjust,
    bootstrap_delta_ci,
    compare_two_methods,
    mcnemar_exact_pvalue,
    two_proportion_ztest_pvalue,
)

RESPONSE_W_THRESH = 0.5  # blend weight counted as "full/meaningful recovery authority"

ARMS = {
    "no_recovery":     "sac_her",
    "v2":              "sac_her_recovery_v2",
    "v3":              "sac_her_recovery_v3",
    "gradual_v4":      "sac_her_recovery_v4",
    "selective_final": "sac_her_recovery_v4_hx6",
}
METHODS = list(ARMS.values())
BASELINE_METHOD = "sac_her"

EP_FILENAME = "episode_results_sac_her_pickandplace_clean_2M.csv"
STEP_FILENAME = "step_logs_sac_her_pickandplace_clean_2M.csv"

# Every directory holding episode_results/step_logs for at least one of
# METHODS at either seed range (0-4 or 5-14) -- see session_handoff_10.md /
# audit_recovery_do_no_harm.py's own DATA_SOURCES for the pre-existing ones;
# results/data_recovery_v4_v2v3_backfill is the seeds-5-14 backfill that
# brought v2/v3 up to n=450; results/data_recovery_v4_hx6 is hx6's own
# full-power evaluation (2026-07-23, the adopted final controller).
DATA_SOURCES = [
    "results/data_recovery_v4",
    "results/data_recovery_v4_hx6",
    "results/data_recovery_v4_power_check",
    "results/data_recovery_v4_power_check_round2",
    "results/data_recovery_v4_power_check_sacher_backfill",
    "results/data_recovery_v4_v2v3_backfill",
]

V4_FAMILY_METHODS = {"sac_her_recovery_v4", "sac_her_recovery_v4_hx6"}


def onset_step(condition: str) -> int:
    return GOAL_SPOOF_MIDEP_STEP if condition == "goal_spoof_midep" else 0


# ---------------------------------------------------------------------------
# Episode-level metrics (1, 2, 6, 8)
# ---------------------------------------------------------------------------

def load_episode_data() -> pd.DataFrame:
    frames = []
    for src in DATA_SOURCES:
        path = os.path.join(src, EP_FILENAME)
        if not os.path.exists(path):
            print(f"[final] WARNING: missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        df = df[df["method"].isin(METHODS)]
        df["_source"] = src
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["method", "condition", "seed"]).reset_index(drop=True)
    combined["episode_in_seed"] = combined.groupby(["method", "condition", "seed"]).cumcount()
    dup_key = ["method", "condition", "seed", "episode_in_seed"]
    n_dupes = combined.duplicated(subset=dup_key).sum()
    if n_dupes:
        raise ValueError(
            f"[final] {n_dupes} duplicate (method, condition, seed, episode_in_seed) "
            "rows found across data sources -- investigate before trusting results."
        )
    return combined


def compare_metric(df: pd.DataFrame, method: str, baseline_method: str, condition: str,
                    metric_col: str, binary: bool, restrict_success: bool = False) -> dict:
    """Generalized version of audit_recovery_do_no_harm.compare_two_methods --
    works on any per-episode metric column, not just 'success'. binary=True
    reuses the audit script's own McNemar/two-proportion-z tests; binary=False
    uses Wilcoxon signed-rank (paired) / Mann-Whitney U (unpaired) instead,
    since McNemar only applies to 0/1 outcomes.
    """
    base = df[(df["method"] == baseline_method) & (df["condition"] == condition)]
    rec = df[(df["method"] == method) & (df["condition"] == condition)]
    if restrict_success:
        base = base[base["success"] == 1]
        rec = rec[rec["success"] == 1]
    base = base.dropna(subset=[metric_col])
    rec = rec.dropna(subset=[metric_col])

    if len(base) == 0 or len(rec) == 0:
        return None

    base_seeds = set(base["seed"].unique())
    rec_seeds = set(rec["seed"].unique())
    shared_seeds = base_seeds & rec_seeds

    rec_mean = rec[metric_col].mean()
    base_mean = base[metric_col].mean()
    delta = rec_mean - base_mean

    paired = shared_seeds == base_seeds == rec_seeds
    if shared_seeds:
        base_p = base[base["seed"].isin(shared_seeds)].set_index(["seed", "episode_in_seed"])[metric_col]
        rec_p = rec[rec["seed"].isin(shared_seeds)].set_index(["seed", "episode_in_seed"])[metric_col]
        common_idx = base_p.index.intersection(rec_p.index)
        base_p = base_p.loc[common_idx].sort_index()
        rec_p = rec_p.loc[common_idx].sort_index()
    else:
        base_p = rec_p = None

    fully_paired = (
        base_p is not None and paired
        and len(base_p) == len(base) and len(rec_p) == len(rec) and len(base_p) > 0
    )

    if fully_paired:
        b_arr = base_p.values
        r_arr = rec_p.values
        ci_lo, ci_hi = bootstrap_delta_ci(r_arr, b_arr, paired=True)
        if binary:
            b10 = int(((r_arr == 1) & (b_arr == 0)).sum())
            b01 = int(((r_arr == 0) & (b_arr == 1)).sum())
            p_value = mcnemar_exact_pvalue(b01, b10)
            test_type = "paired_mcnemar"
        else:
            diffs = r_arr - b_arr
            if np.all(diffs == 0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(r_arr, b_arr, zero_method="wilcox").pvalue)
            test_type = "paired_wilcoxon"
        n_base, n_rec = len(b_arr), len(r_arr)
    else:
        ci_lo, ci_hi = bootstrap_delta_ci(rec[metric_col].values, base[metric_col].values, paired=False)
        if binary:
            count = [int(rec[metric_col].sum()), int(base[metric_col].sum())]
            nobs = [len(rec), len(base)]
            p_value = two_proportion_ztest_pvalue(count, nobs)
            test_type = "unpaired_two_prop_z"
        else:
            p_value = float(mannwhitneyu(rec[metric_col].values, base[metric_col].values,
                                          alternative="two-sided").pvalue)
            test_type = "unpaired_mannwhitney"
        n_base, n_rec = len(base), len(rec)

    return {
        "method": method,
        "baseline_method": baseline_method,
        "condition": condition,
        "metric": metric_col,
        "n_baseline": n_base,
        "n_recovery": n_rec,
        "baseline_mean": base_mean,
        "recovery_mean": rec_mean,
        "delta": delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "test_type": test_type,
    }


# ---------------------------------------------------------------------------
# Step-level metrics (3, 4, 5, 7)
# ---------------------------------------------------------------------------

STEP_USECOLS = ["method", "condition", "seed", "episode_idx", "timestep",
                 "recovery_triggered", "recovery_v4_weight"]


def _reduce_step_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=STEP_USECOLS)
    df = df[df["method"].isin(METHODS)]
    if len(df) == 0:
        return pd.DataFrame()
    df = df.sort_values(["method", "condition", "seed", "episode_idx", "timestep"])

    rows = []
    group_cols = ["method", "condition", "seed", "episode_idx"]
    for (method, condition, seed, ep_idx), g in df.groupby(group_cols, sort=False):
        onset = onset_step(condition)
        g = g[g["timestep"] >= onset]
        triggered = g["recovery_triggered"].fillna(0).astype(int).values
        timesteps = g["timestep"].values

        trig_idx = np.flatnonzero(triggered == 1)
        if len(trig_idx) == 0:
            detection_delay = np.nan
            response_delay = np.nan
        else:
            first_trigger_t = timesteps[trig_idx[0]]
            detection_delay = float(first_trigger_t - onset)
            if method in V4_FAMILY_METHODS:
                w = g["recovery_v4_weight"].fillna(0.0).values
                post = np.flatnonzero((w >= RESPONSE_W_THRESH) & (timesteps >= first_trigger_t))
                response_delay = float(timesteps[post[0]] - first_trigger_t) if len(post) else np.nan
            else:
                response_delay = 0.0  # v2/v3: full unblended override on the trigger step itself

        recovery_time = int(triggered.sum())
        rising_edges = int(np.sum((triggered[1:] == 1) & (triggered[:-1] == 0)) + (triggered[0] == 1))

        rows.append({
            "method": method, "condition": condition, "seed": seed, "episode_idx": ep_idx,
            "detection_delay": detection_delay, "response_delay": response_delay,
            "recovery_time": recovery_time, "num_interventions": rising_edges,
        })
    return pd.DataFrame(rows)


def load_step_metrics() -> pd.DataFrame:
    frames = []
    for src in DATA_SOURCES:
        path = os.path.join(src, STEP_FILENAME)
        if not os.path.exists(path):
            print(f"[final] WARNING: missing {path}, skipping step-level metrics for this source")
            continue
        print(f"[final] Reducing step log: {path}")
        frames.append(_reduce_step_log(path))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ep = load_episode_data()
    print(f"[final] Loaded {len(ep)} episodes across {len(METHODS)} methods.")
    coverage = ep.groupby(["method", "condition"])["seed"].nunique().unstack(fill_value=0)
    print("\n[final] Seed coverage per method x condition (every cell should read 15):")
    print(coverage.to_string())

    step_metrics = load_step_metrics()
    # episode_in_seed on the step-derived table, to align with episode_results' pairing key.
    step_metrics = step_metrics.sort_values(["method", "condition", "seed", "episode_idx"])
    step_metrics["episode_in_seed"] = step_metrics.groupby(
        ["method", "condition", "seed"]).cumcount()
    merged = ep.merge(
        step_metrics.drop(columns=["episode_idx"]),
        on=["method", "condition", "seed", "episode_in_seed"], how="left",
    )

    # -- Metrics 1/2: task-success rate + clean-task performance -------------
    success_rows = []
    for method in ARMS.values():
        if method == BASELINE_METHOD:
            continue
        for condition in ALL_CONDITIONS:
            row = compare_two_methods(merged, method, BASELINE_METHOD, condition)
            if row is not None:
                success_rows.append(row)
    success_df = pd.DataFrame(success_rows)
    success_df["p_value_bh"] = bh_adjust(success_df["p_value"].values)
    success_df["significant_bh"] = success_df["p_value_bh"] < 0.05

    # -- Metric 6: safety violations -----------------------------------------
    safety_rows = []
    for method in ARMS.values():
        if method == BASELINE_METHOD:
            continue
        for condition in ALL_CONDITIONS:
            row = compare_metric(merged, method, BASELINE_METHOD, condition,
                                  "safety_violation", binary=True)
            if row is not None:
                safety_rows.append(row)
    safety_df = pd.DataFrame(safety_rows)
    safety_df["p_value_bh"] = bh_adjust(safety_df["p_value"].values)
    safety_df["significant_bh"] = safety_df["p_value_bh"] < 0.05

    success_safety = pd.concat([
        success_df.assign(metric="task_success_rate"),
        safety_df,
    ], ignore_index=True)
    out1 = "results/final_hx_comparison_success_safety.csv"
    success_safety.to_csv(out1, index=False)
    print(f"\n[final] Wrote {len(success_safety)} rows to {out1}")

    # -- Head-to-head: selective_final (the adopted final controller) vs --
    # each other arm directly, not just each arm vs sac_her. This is the
    # comparison the mentor's "final vs earlier baselines" framing actually
    # calls for (e.g. hx6 vs plain v4 on grip_state_falsification is the
    # already-documented +4.2pp/p=0.0017 win, inherited unchanged from
    # v4_hx2 -- vs-sac_her alone shows "parity", which is a different,
    # also-true, but distinct claim). Own BH scope, separate from the
    # vs-sac_her grid above (distinct family of comparisons, same
    # precedent as scripts/evaluate_recovery_v4_hx5.py treating "vs
    # v4/hx2" and "vs sac_her" as separately-corrected blocks).
    h2h_rows = []
    for baseline in ["sac_her_recovery_v2", "sac_her_recovery_v3", "sac_her_recovery_v4"]:
        for condition in ALL_CONDITIONS:
            row = compare_metric(merged, "sac_her_recovery_v4_hx6", baseline, condition,
                                  "success", binary=True)
            if row is not None:
                h2h_rows.append(row)
    h2h_df = pd.DataFrame(h2h_rows)
    h2h_df["p_value_bh"] = bh_adjust(h2h_df["p_value"].values)
    h2h_df["significant_bh"] = h2h_df["p_value_bh"] < 0.05
    out1b = "results/final_hx_comparison_final_head_to_head.csv"
    h2h_df.assign(metric="task_success_rate").to_csv(out1b, index=False)
    print(f"[final] Wrote {len(h2h_df)} hx6-vs-other-arms rows to {out1b}")
    sig_h2h = h2h_df[h2h_df.significant_bh]
    print("[final] hx6 (final) vs v2/v3/v4, BH-significant task-success differences:")
    print(sig_h2h[["method", "baseline_method", "condition", "baseline_mean",
                    "recovery_mean", "delta", "p_value_bh"]].to_string(index=False)
          if len(sig_h2h) else "        (none)")

    # -- Metric 8: completion-time overhead (successful episodes only) -------
    overhead_rows = []
    for method in ARMS.values():
        if method == BASELINE_METHOD:
            continue
        for condition in ALL_CONDITIONS:
            row = compare_metric(merged, method, BASELINE_METHOD, condition,
                                  "first_success_step", binary=False, restrict_success=True)
            if row is not None:
                overhead_rows.append(row)
    overhead_df = pd.DataFrame(overhead_rows)
    if len(overhead_df):
        overhead_df["p_value_bh"] = bh_adjust(overhead_df["p_value"].values)
        overhead_df["significant_bh"] = overhead_df["p_value_bh"] < 0.05

    # -- Metrics 5/7: recovery time, interventions -- these ARE meaningful vs
    # sac_her (which is always exactly 0 for both, a valid "doing nothing"
    # baseline, unlike detection/response delay below). Non-clean conditions
    # only (nothing to detect on clean).
    timing_rows = []
    for method in ARMS.values():
        if method == BASELINE_METHOD:
            continue
        for condition in [c for c in ALL_CONDITIONS if c != "clean"]:
            for metric_col in ["recovery_time", "num_interventions"]:
                row = compare_metric(merged, method, BASELINE_METHOD, condition,
                                      metric_col, binary=False)
                if row is not None:
                    timing_rows.append(row)
    timing_df = pd.DataFrame(timing_rows)
    if len(timing_df):
        timing_df["p_value_bh"] = bh_adjust(timing_df["p_value"].values)
        timing_df["significant_bh"] = timing_df["p_value_bh"] < 0.05

    # -- Metrics 3/4: detection delay, recovery-response delay -- NOT
    # compared against sac_her (it never triggers recovery at all, so its
    # delay values are undefined/NaN, not a meaningful 0 -- unlike
    # recovery_time/num_interventions above, there is no "doing nothing"
    # baseline for a delay that never happens). Reported as plain
    # descriptive stats per (method, condition) instead -- mean/median over
    # episodes where that method DID trigger, plus the trigger rate itself
    # (how often it triggered at all), which is the honest denominator.
    delay_rows = []
    for method in ARMS.values():
        if method == BASELINE_METHOD:
            continue
        for condition in [c for c in ALL_CONDITIONS if c != "clean"]:
            sub = merged[(merged.method == method) & (merged.condition == condition)]
            n_episodes = len(sub)
            det = sub["detection_delay"].dropna()
            resp = sub["response_delay"].dropna()
            delay_rows.append({
                "method": method, "condition": condition, "n_episodes": n_episodes,
                "trigger_rate": len(det) / n_episodes if n_episodes else np.nan,
                "detection_delay_mean": det.mean() if len(det) else np.nan,
                "detection_delay_median": det.median() if len(det) else np.nan,
                "response_delay_mean": resp.mean() if len(resp) else np.nan,
                "response_delay_median": resp.median() if len(resp) else np.nan,
            })
    delay_df = pd.DataFrame(delay_rows)
    out_delay = "results/final_hx_comparison_delays.csv"
    delay_df.to_csv(out_delay, index=False)
    print(f"[final] Wrote {len(delay_df)} rows to {out_delay}")

    out2 = "results/final_hx_comparison_timing.csv"
    pd.concat([overhead_df.assign(metric="completion_time_overhead") if len(overhead_df) else pd.DataFrame(),
               timing_df], ignore_index=True).to_csv(out2, index=False)
    print(f"[final] Wrote timing/overhead comparisons to {out2}")

    print("\n[final] Summary means by method (non-clean conditions, all metrics):")
    summary_cols = ["detection_delay", "response_delay", "recovery_time", "num_interventions"]
    print(merged[merged.condition != "clean"].groupby("method")[summary_cols].mean().to_string())
    print("\n[final] Trigger rate by method (fraction of non-clean episodes that triggered at all):")
    print(delay_df.groupby("method")["trigger_rate"].mean().to_string())


if __name__ == "__main__":
    main()
