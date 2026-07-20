"""
Item 1 Phase 1b (step 3) — Finalize Level 2 (anomaly detection) categorical
labels from the Phase 1a Candidate D p_fail scores.

Decision locked in 2026-07-20 (see CLAUDE.md "Level 2 (Anomaly Detection)
Scope Limitation"):

  - Level 2's categorical output is scoped to the clean_2M checkpoint only.
    The other three checkpoints (clean_500k, randomized_2M, randomized_500k)
    have no discriminative "normal" band even on their own clean episodes
    (Phase 1a finding) -> every row from those checkpoints is `unknown`.
  - Within clean_2M, early-checkpoint rows (t<69) are ALSO `unknown` --
    Phase 1a found these reflect task-progression ambiguity (mean p_fail
    ~0.65 at t=19-59 vs ~0.15/median 0.005 at t>=69), not genuine anomaly
    signal, and are kept separate from suspicious/abnormal per this round's
    decision.
  - normal/suspicious/abnormal cut points are the clean_2M, t>=69, clean-only
    p75/p95 computed in Phase 1a (exact values below, not rounded).

Input:  results/level2_pfail_scores.csv        (Phase 1a, unmodified)
Output: results/level2_labels_clean_2M_scoped.csv
"""

import numpy as np
import pandas as pd

IN_PATH = "results/level2_pfail_scores.csv"
OUT_PATH = "results/level2_labels_clean_2M_scoped.csv"

EARLY_CUTOFF_T = 69  # t < 69 -> unknown (task-progression artifact, clean_2M only)

df = pd.read_csv(IN_PATH)

# ── Exact clean_2M, clean-condition p75/p95 — the same numbers reported in the
# Phase 1a B2 percentile table (pooled across all 14 checkpoints of the clean
# condition, model==clean_2M). Recomputed here bit-for-bit from the Phase 1a
# level2_pfail_scores.csv rather than hardcoded, so the exact values are
# reproducible and auditable. NOT restricted to t>=69 -- that restriction is
# applied only to which rows RECEIVE a normal/suspicious/abnormal verdict
# (see _label below), not to how the cut points themselves were calibrated.
clean_ref = df[(df["model"] == "clean_2M") & (df["condition"] == "clean")]["p_fail"].values
CLEAN_P75 = float(np.percentile(clean_ref, 75))
CLEAN_P95 = float(np.percentile(clean_ref, 95))
print(f"[level2-finalize] clean_2M, clean-only reference, all checkpoints pooled (n={len(clean_ref)}):")
print(f"  CLEAN_P75 = {CLEAN_P75!r}")
print(f"  CLEAN_P95 = {CLEAN_P95!r}\n")


def _label(row):
    if row["model"] != "clean_2M":
        return "unknown"
    if row["checkpoint_t"] < EARLY_CUTOFF_T:
        return "unknown"
    if row["p_fail"] < CLEAN_P75:
        return "normal"
    if row["p_fail"] < CLEAN_P95:
        return "suspicious"
    return "abnormal"


df["level2_label"] = df.apply(_label, axis=1)
df["level2_threshold_p75"] = CLEAN_P75
df["level2_threshold_p95"] = CLEAN_P95

df.to_csv(OUT_PATH, index=False)
print(f"[level2-finalize] Saved -> {OUT_PATH}  ({len(df)} rows)\n")

print("=" * 70)
print("CLASS DISTRIBUTION (all rows, all models/conditions/checkpoints)")
print("=" * 70)
counts = df["level2_label"].value_counts()
pct = (100 * counts / len(df)).round(2)
dist_df = pd.DataFrame({"count": counts, "pct": pct})
print(dist_df.to_string())
print()

print("=" * 70)
print("CLASS DISTRIBUTION WITHIN clean_2M ONLY (23,100 rows)")
print("=" * 70)
c2m = df[df["model"] == "clean_2M"]
counts2 = c2m["level2_label"].value_counts()
pct2 = (100 * counts2 / len(c2m)).round(2)
dist_df2 = pd.DataFrame({"count": counts2, "pct": pct2})
print(dist_df2.to_string())
print()

print("=" * 70)
print("CLASS DISTRIBUTION WITHIN clean_2M, t>=69 ONLY (i.e. normal/suspicious/abnormal candidates)")
print("=" * 70)
c2m_late = c2m[c2m["checkpoint_t"] >= EARLY_CUTOFF_T]
counts3 = c2m_late["level2_label"].value_counts()
pct3 = (100 * counts3 / len(c2m_late)).round(2)
dist_df3 = pd.DataFrame({"count": counts3, "pct": pct3})
print(dist_df3.to_string())

print("\n[level2-finalize] Done.")
