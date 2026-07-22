"""
Do-no-harm audit for TAIRO PickAndPlace recovery methods (clean_2M checkpoint).

Systematically compares every recovery method (sac_her_recovery_v2/v3/v4/
v4_hx/v4_hx2) against the plain sac_her baseline, across all 11 conditions,
using whatever episode data is available in the directories below. Flags
any (method, condition) pair where recovery does WORSE than doing nothing.

Data sources (episode_results_sac_her_pickandplace_clean_2M.csv in each):
    results/data_recovery_v4/                        seeds 0-4,  sac_her + v2 + v3 + v4, all 11 conditions
    results/data_recovery_v4_hx/                      seeds 0-4,  v4_hx,  all 11 conditions
    results/data_recovery_v4_hx2/                     seeds 0-4,  v4_hx2, all 11 conditions
    results/data_recovery_v4_power_check/             seeds 5-14, v4 + v4_hx + v4_hx2, 4 conditions
    results/data_recovery_v4_power_check_round2/      seeds 5-14, sac_her + v4 + v4_hx + v4_hx2, 7 conditions
    results/data_recovery_v4_power_check_sacher_backfill/  seeds 5-14, sac_her, 4 conditions

Episodes are paired by (condition, seed, episode_in_seed) -- reconstructed
via a per-(condition, method, seed) cumulative count, since
evaluation/episode_runner.py sets reset_seed = 100*seed + episode_in_seed,
so the same (seed, episode_in_seed) pair is the identical physical episode
across methods (confirmed in RECOVERY_V4.md / findings.md Phase 11).

Where both arms have the same seed set, a paired McNemar test is used
(statsmodels, exact binomial for small discordant counts). Where seed sets
differ, an unpaired two-proportion bootstrap CI is used instead. Both cases
also report a percentile bootstrap CI on the success-rate delta.

Multiple-testing correction: Benjamini-Hochberg FDR across all method x
condition comparisons against sac_her (confirmed via sign-off 2026-07-21 --
55 comparisons in a systematic scan warrants correction, unlike the 2
targeted conditions in RECOVERY_V4.md section 5.3's ad hoc test).

Output: results/recovery_do_no_harm_audit.csv (one row per method x
condition), plus a printed summary of flagged harm cases.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from config import ALL_CONDITIONS

N_BOOT = 5000
RNG_SEED = 42

DATA_SOURCES = [
    "results/data_recovery_v4",
    "results/data_recovery_v4_hx",
    "results/data_recovery_v4_hx2",
    "results/data_recovery_v4_power_check",
    "results/data_recovery_v4_power_check_round2",
    "results/data_recovery_v4_power_check_sacher_backfill",
]
EP_FILENAME = "episode_results_sac_her_pickandplace_clean_2M.csv"
BASELINE_METHOD = "sac_her"
RECOVERY_METHODS = [
    "sac_her_recovery_v2",
    "sac_her_recovery_v3",
    "sac_her_recovery_v4",
    "sac_her_recovery_v4_hx",
    "sac_her_recovery_v4_hx2",
]


def load_all_episodes() -> pd.DataFrame:
    frames = []
    for src in DATA_SOURCES:
        path = os.path.join(src, EP_FILENAME)
        if not os.path.exists(path):
            print(f"[audit] WARNING: missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        df["_source"] = src
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Reconstruct episode_in_seed: episode_idx is a global monotonic counter
    # (not stable across separate sweep invocations), so re-derive the
    # within-(method, condition, seed) episode index directly.
    combined = combined.sort_values(["method", "condition", "seed"]).reset_index(drop=True)
    combined["episode_in_seed"] = combined.groupby(["method", "condition", "seed"]).cumcount()

    # De-duplicate: if any (method, condition, seed, episode_in_seed) row
    # appears more than once across data sources (shouldn't happen given
    # the disjoint seed/condition partition each sweep was run with, but
    # verify rather than assume).
    dup_key = ["method", "condition", "seed", "episode_in_seed"]
    n_dupes = combined.duplicated(subset=dup_key).sum()
    if n_dupes:
        raise ValueError(
            f"[audit] {n_dupes} duplicate (method, condition, seed, episode_in_seed) "
            "rows found across data sources -- investigate before trusting results."
        )
    return combined


def mcnemar_exact_pvalue(b01: int, b10: int) -> float:
    """Exact McNemar test p-value (two-sided binomial on discordant pairs)."""
    n_discordant = b01 + b10
    if n_discordant == 0:
        return 1.0
    k = min(b01, b10)
    return float(binomtest(k, n_discordant, 0.5, alternative="two-sided").pvalue)


def two_proportion_ztest_pvalue(count: list, nobs: list) -> float:
    x1, x2 = count
    n1, n2 = nobs
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return float(2 * (1 - norm.cdf(abs(z))))


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj_ranked = np.empty(n)
    prev_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev_min = min(prev_min, val)
        adj_ranked[i] = prev_min
    adj = np.empty(n)
    adj[order] = adj_ranked
    return adj


def bootstrap_delta_ci(success_a: np.ndarray, success_b: np.ndarray, paired: bool,
                        n_boot: int = N_BOOT, seed: int = RNG_SEED):
    """95% percentile bootstrap CI on mean(success_a) - mean(success_b).

    paired=True resamples matched pairs jointly (same episode-pair index
    each draw); paired=False resamples each arm independently.
    """
    rng = np.random.default_rng(seed)
    n_a = len(success_a)
    deltas = np.empty(n_boot)
    if paired:
        assert len(success_a) == len(success_b)
        for i in range(n_boot):
            idx = rng.integers(0, n_a, n_a)
            deltas[i] = success_a[idx].mean() - success_b[idx].mean()
    else:
        n_b = len(success_b)
        for i in range(n_boot):
            idx_a = rng.integers(0, n_a, n_a)
            idx_b = rng.integers(0, n_b, n_b)
            deltas[i] = success_a[idx_a].mean() - success_b[idx_b].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def compare_two_methods(df: pd.DataFrame, method: str, baseline_method: str, condition: str) -> dict:
    base = df[(df["method"] == baseline_method) & (df["condition"] == condition)]
    rec = df[(df["method"] == method) & (df["condition"] == condition)]

    if len(base) == 0 or len(rec) == 0:
        return None

    base_seeds = set(base["seed"].unique())
    rec_seeds = set(rec["seed"].unique())
    shared_seeds = base_seeds & rec_seeds

    rec_rate = rec["success"].mean()
    base_rate = base["success"].mean()
    delta = rec_rate - base_rate

    # Paired subset (same seed set) drives McNemar; if the seed sets are
    # fully identical, the paired subset equals the full data on both sides.
    paired = shared_seeds == base_seeds == rec_seeds
    if shared_seeds:
        base_p = base[base["seed"].isin(shared_seeds)].set_index(["seed", "episode_in_seed"])["success"]
        rec_p = rec[rec["seed"].isin(shared_seeds)].set_index(["seed", "episode_in_seed"])["success"]
        common_idx = base_p.index.intersection(rec_p.index)
        base_p = base_p.loc[common_idx].sort_index()
        rec_p = rec_p.loc[common_idx].sort_index()
    else:
        base_p = rec_p = None

    if base_p is not None and len(base_p) == len(base) and len(rec_p) == len(rec):
        # Fully paired (identical seed sets, every episode matched) -> McNemar + paired bootstrap.
        b_arr = base_p.values
        r_arr = rec_p.values
        # Discordant pairs: recovery succeeds but baseline fails (b01) / vice versa (b10)
        b10 = int(((r_arr == 1) & (b_arr == 0)).sum())  # recovery helped
        b01 = int(((r_arr == 0) & (b_arr == 1)).sum())  # recovery hurt
        p_value = mcnemar_exact_pvalue(b01, b10)
        ci_lo, ci_hi = bootstrap_delta_ci(r_arr, b_arr, paired=True)
        test_type = "paired_mcnemar"
        n_base, n_rec = len(b_arr), len(r_arr)
    else:
        # Seed sets differ (or partial overlap) -> unpaired two-sample bootstrap.
        ci_lo, ci_hi = bootstrap_delta_ci(rec["success"].values, base["success"].values, paired=False)
        count = [int(rec["success"].sum()), int(base["success"].sum())]
        nobs = [len(rec), len(base)]
        p_value = two_proportion_ztest_pvalue(count, nobs)
        test_type = "unpaired_bootstrap"
        n_base, n_rec = len(base), len(rec)

    return {
        "method": method,
        "baseline_method": baseline_method,
        "condition": condition,
        "n_baseline": n_base,
        "n_recovery": n_rec,
        "baseline_success_rate": base_rate,
        "recovery_success_rate": rec_rate,
        "delta": delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "test_type": test_type,
    }


def main():
    combined = load_all_episodes()
    print(f"[audit] Loaded {len(combined)} episodes from {len(DATA_SOURCES)} sources.")

    coverage = combined.groupby(["method", "condition"])["seed"].nunique().unstack(fill_value=0)
    print("\n[audit] Seed coverage per method x condition:")
    print(coverage.to_string())

    rows = []
    for method in RECOVERY_METHODS:
        for condition in ALL_CONDITIONS:
            row = compare_two_methods(combined, method, BASELINE_METHOD, condition)
            if row is not None:
                rows.append(row)

    result_df = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction across all comparisons in this audit.
    p_adj = bh_adjust(result_df["p_value"].values)
    result_df["p_value_bh"] = p_adj
    result_df["significant_bh"] = p_adj < 0.05

    result_df["harm_flag"] = (
        (result_df["delta"] < 0) & result_df["significant_bh"]
    )

    result_df = result_df.sort_values(["harm_flag", "delta"], ascending=[False, True])

    out_path = "results/recovery_do_no_harm_audit.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\n[audit] Wrote {len(result_df)} comparisons to {out_path}")

    harm = result_df[result_df["harm_flag"]]
    print(f"\n[audit] {len(harm)} of {len(result_df)} comparisons show SIGNIFICANT HARM "
          f"(recovery worse than sac_her, BH-adjusted p<0.05):")
    if len(harm):
        print(harm[["method", "condition", "baseline_success_rate", "recovery_success_rate",
                     "delta", "ci_lo", "ci_hi", "p_value", "p_value_bh", "test_type"]]
              .to_string(index=False))
    else:
        print("        (none)")

    benefit = result_df[(result_df["delta"] > 0) & result_df["significant_bh"]]
    print(f"\n[audit] {len(benefit)} comparisons show SIGNIFICANT BENEFIT (BH-adjusted p<0.05):")
    if len(benefit):
        print(benefit[["method", "condition", "baseline_success_rate", "recovery_success_rate",
                        "delta", "ci_lo", "ci_hi", "p_value", "p_value_bh", "test_type"]]
              .to_string(index=False))
    else:
        print("        (none)")


if __name__ == "__main__":
    main()
