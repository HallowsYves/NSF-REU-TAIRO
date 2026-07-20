# Investigation: Level 3 Multi-Label Feasibility + Taxonomy Label Provenance

**Branch:** `investigate/level3-multilabel-and-taxonomy` (off `main`)
**Type:** Phase 0, read-only. No training, no label changes, no commits.
**Date:** 2026-07-20

---

## Question 2 first — How were the existing failure-type labels generated?

(Answered first because it's the load-bearing question — the answer to Q1 is interpreted
differently depending on it.)

### Answer: behavior-based, not attack-based. Confirmed directly in code, not just docs.

**Evidence 1 — the labeler's own docstring and code never touch attack identity.**

`evaluation/failure_mode_labeling.py:8-9`:
```
Returns one of six labels based on trajectory behavior alone — the
`condition` and `attack_level` columns are never used as features.
```

This isn't just a doc claim — I read the full body of `label_episode()`
(`evaluation/failure_mode_labeling.py:161-229`) and it only ever reads these columns:
`distance_to_object`, `distance_to_true_goal`, `distance_to_perceived_goal`, `is_success`,
`object_velp_*`, `grip_velp_*`, `object_pos_z`. There is no reference to `condition` or
`attack_level` anywhere in the function body — the label is a pure function of kinematic
trajectory (gripper/object distances, grasp kinematics + lift confirmation, drop detection,
goal convergence), computed identically regardless of which attack (if any) produced that
trajectory.

**Evidence 2 — the causal/online feature builder enforces this as a hard constraint, with a
runtime assertion, not just a comment.**

`evaluation/causal_features.py:42-45`:
```
Hard constraint
---------------
`condition` and `attack_level` are never features — same discipline as
Phase 8 (`scripts/train_failure_classifier.py`).
```
`evaluation/causal_features.py:66`: `FORBIDDEN_FEATURES = {"condition", "attack_level", "method"}`

`scripts/build_causal_feature_matrix.py:77-78`:
```python
assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES), \
    f"Forbidden feature leaked: {set(FEATURE_COLS) & FORBIDDEN_FEATURES}"
```
This assertion runs every time the feature matrix is built — it's a hard guard against
attack-identity leaking into what feeds the classifier, not an aspirational statement.

**Evidence 3 — concrete many-to-many mapping, computed directly from the labeled data
(`results/data_seedfix/sac_her_only/labels_sac_her_pickandplace_all.csv`, 6,600 episodes,
all 4 models × 11 conditions):**

```
action_clipping           never_reached_object:301, success:168, reached_but_failed_grasp:119, grasped_but_dropped:7,  divergent_transport:5
action_delay               never_reached_object:293, reached_but_failed_grasp:144, success:131, grasped_but_dropped:21, divergent_transport:11
action_reversal            never_reached_object:587, reached_but_failed_grasp:12,  divergent_transport:1
clean                      never_reached_object:305, success:155, reached_but_failed_grasp:113, grasped_but_dropped:18, divergent_transport:9
contact_dropout            never_reached_object:583, reached_but_failed_grasp:16,  grasped_but_dropped:1
goal_spoof_immediate       never_reached_object:325, spoofed_goal:138, reached_but_failed_grasp:106, grasped_but_dropped:18, success:11, divergent_transport:2
goal_spoof_midep           never_reached_object:296, spoofed_goal:157, reached_but_failed_grasp:98,  grasped_but_dropped:25, success:23, divergent_transport:1
grip_state_falsification   never_reached_object:338, reached_but_failed_grasp:191, divergent_transport:33, success:28,  grasped_but_dropped:10
object_pose_spoof          never_reached_object:312, reached_but_failed_grasp:132, divergent_transport:106, success:31,  grasped_but_dropped:19
sensor_bias                never_reached_object:520, reached_but_failed_grasp:62,  divergent_transport:10, grasped_but_dropped:6, success:2
sensor_dropout             never_reached_object:580, reached_but_failed_grasp:19,  grasped_but_dropped:1
```

This is decisive evidence against a 1:1 attack↔label mapping:
- **No condition maps to a single label.** Every one of the 11 conditions — including
  `clean` — produces a spread across 3-6 of the 6 labels.
- **`spoofed_goal` is not exclusive to goal-spoof attacks.** It appears under
  `goal_spoof_immediate` and `goal_spoof_midep` as expected, but is (correctly) absent
  everywhere else — this is actually a case where a label *does* track one attack family
  fairly tightly, but it's a consequence of the kinematic definition (converges to perceived
  goal while staying far from true goal) coinciding with what that attack produces, not
  because the label reads attack identity.
- **`never_reached_object` and `reached_but_failed_grasp` dominate almost every condition**,
  including `clean` (305 and 113 respectively, out of 600) — i.e., the same behavioral
  failure mode is the majority outcome under both benign and adversarial conditions. If
  labeling were attack-derived, `clean` episodes could not produce non-`success` labels at
  all (there is no attack to attribute them to), yet 445/600 clean episodes carry a non-success
  label purely from behavior.
- **`divergent_transport` under `object_pose_spoof` (106/600) vs. under `grip_state_falsification`
  (33/600) vs. under `clean` (9/600)** — one label, three structurally different root causes,
  distinguished only by what the trajectory did, not by what attack was active.

**Conclusion:** the existing six-label taxonomy is genuinely symptom/behavior-derived. Reusing
it for Level 3 (Failure Type) does **not** collapse it into Level 4 (Attack Family) — the two
are already empirically decorrelated in the existing data. This removes the main risk flagged
in the task's "why this matters" section: reuse is not a design regression here.

---

## Question 1 — Is multi-label feasible with current data, without new data generation?

### Short answer: structurally, the pipeline is single-label end-to-end today — but the
underlying signal needed for multi-label already exists in derived data that's currently on
disk, or one script has to be re-run to get the full-fidelity version. It is not a new
data-generation problem in either case; it is a re-labeling / re-derivation problem.

**Structurally single-label, by construction, in two places:**

1. `label_episode()` returns `-> str`, a single label, and is implemented as an if/elif chain
   with early returns (`evaluation/failure_mode_labeling.py:161-229`) — steps 1 through 6 are
   checked in priority order and the function exits at the first match. Even though several of
   the underlying boolean conditions (never-reached, no-grasp, dropped, spoofed-goal-active,
   diverging) are computed as independent checks internally, the function currently keeps only
   the first true one and discards the rest.
2. `label_batch()` (`evaluation/failure_mode_labeling.py:232-247`) produces one row per
   `episode_idx` with a single `failure_mode` column — this is the shape every downstream
   consumer (`train_failure_classifier.py`, `build_causal_feature_matrix.py`,
   `train_causal_classifier.py`, `build_online_classifier.py`) expects.

**But the raw signal for co-occurring symptoms already exists, at two different levels of
completeness, and neither requires generating new episodes:**

- **Already on disk right now:** `results/classifier_seedfix/causal_feature_matrix.csv`
  (92,401 rows, 13-14 checkpoints/episode across all 6,600 episodes) carries per-checkpoint
  continuous features that are exactly the kind of signal multi-label would need —
  `separated_after_contact_now` (drop signal), `goal_offset_now`/`goal_offset_max_sofar`
  (spoof signal), `safety_violation_rate_sofar`, `dttg_slope_long` (divergence signal),
  `grasp_kinematic_ever_sofar`, `reached_ever_sofar`, etc. (full column list confirmed by
  reading the header). Because these are independent continuous/boolean quantities rather
  than a single collapsed label, you could derive approximate multi-label symptom flags
  (e.g. "goal_offset elevated AND separated_after_contact true in the same episode") directly
  from this file with no re-run of anything. Caveat: this file uses a sparse 10-step stride
  (`CAUSAL_CHECKPOINT_START=19`, `CAUSAL_CHECKPOINT_STRIDE=10`) and the causal grasp proxy is
  intentionally looser than the hindsight labeler's (~4.6% disagreement, documented in
  `causal_features.py:144-160`), so this route gives an approximate, not exact, multi-label
  derivation.
- **Exact-fidelity version requires regenerating (not newly generating) `step_logs_*.csv`.**
  The full per-step logs that `label_episode()` itself consumes
  (`step_logs_sac_her_pickandplace_{model}.csv`, expected at `DATA_DIR` per
  `scripts/build_causal_feature_matrix.py:52`) are **not currently present** in
  `results/data_seedfix/` — I checked; only aggregate `episode_results_*.csv` and per-episode
  `labels_*.csv` (single `failure_mode` column) survive there. Full per-step logs currently
  exist on disk only for the unrelated Recovery v4 runs
  (`results/data_recovery_v4/step_logs_sac_her_pickandplace_clean_2M.csv` and the `_dense`
  variant — `clean_2M` only, not all 4 models) and one archived diagnostic file. Getting
  exact-fidelity multi-label data for all 4 models would mean re-running the existing
  seed-fixed sweep to regenerate `step_logs_*.csv` — same trained policies, same fixed seeds,
  same 11 conditions, fully deterministic — which reproduces an artifact that already existed
  transiently during the original run but wasn't persisted to disk. This is not new
  data-generation in the sense of new experiments or new attack conditions, but it is a rerun,
  which the task scope asked me to flag rather than execute.

**Rough work estimate (if the design direction is confirmed as worth pursuing):**

- Rewriting `label_episode()`'s internal priority chain into independent boolean symptom
  flags (the checks already exist as separable logic — `_detect_grasp`, `_detect_drop`,
  goal-spoof condition, `_is_diverging`) — a few hours; the hard classification logic is
  already written and tested, this is a refactor of the return shape, not new algorithm design.
- Deriving approximate multi-label from the existing `causal_feature_matrix.csv` — a few
  hours (thresholding existing columns, no new computation).
- Re-running the sweep to regenerate `step_logs_*.csv` for exact-fidelity multi-label across
  all 4 models — this is the one step that isn't pure desk work; order-of-magnitude is
  "a sweep run," not new data collection, but I did not attempt or time it (out of scope here).
- None of this requires new episodes, new attacks, or new simulator runs beyond that one
  optional regeneration step.

---

## Recommendation

Given the findings:

- **The core risk motivating "worst symptom wins" — that reusing the existing taxonomy would
  collapse Level 3 into Level 4 — is not supported by the data.** Labels are behavior-derived
  and already show no meaningful correlation with attack identity (see the crosstab above).
  **Reuse existing taxonomy** looks like the safer and cheaper choice for Level 3 this week,
  not a regression.
- **Flagging multi-label as a third option, per the task's instruction:** it turned out
  cheaper than "structurally infeasible without new data" — the underlying boolean/continuous
  symptom signals already exist in code (`_detect_grasp`, `_detect_drop`, goal-spoof check,
  `_is_diverging`) and, approximately, in an already-on-disk feature matrix. If there's any
  appetite to revisit the memo's original multi-label intent, it's a few-hours refactor, not a
  new data-generation project. I'd flag this to whoever makes the Level 3 call rather than
  assume it's off the table for schedule reasons.
- I have not made this decision — this is the human/mentor call per the task instructions.

---

## Validation Cost Scoping (2026-07-20, appended)

**Task:** scope — not perform — the cost of validating multi-label Level 3 symptoms to a
paper-citable standard, using the existing Level 1 false-alarm bootstrap-CI work as the
calibration reference. No derivation, no rerun, no validation executed.

**Reference point used for calibration:** `retrain/causal-flat-rf-baseline` branch,
`scripts/bootstrap_false_alarm_ci.py` (160 lines) and `scripts/train_and_save_causal_pooled.py`
(177 lines) — findings.md §"False-Alarm Rate — Bootstrap CIs + Early/Mid/Late Binning"
(2026-07-19 to 2026-07-20). This effort trained one pooled RF once, then reused the saved
model + existing `causal_feature_matrix.csv` to compute detection-delay tables, a false-alarm
rate on a single population subset (clean+success, n=420, ~30/checkpoint), and 5,000-resample
percentile bootstrap CIs, binned into early/mid/late (n=120–150/bin) because per-checkpoint
n=30 CIs were too wide to cite individually. Total effort spanned two sessions across two days:
session A (pooled model + detection-delay table + raw false-alarm rate) and session B (the
bootstrap-CI script itself, ~160 lines, reusing the session-A model with no retraining). Session
B — the part directly comparable to "add a bootstrap CI to an existing metric" — is the
realistic per-unit cost to calibrate against: on the order of **half a day** for one metric on
one population subset, most of which is script-writing, not compute (5,000 resamples on
n≤420 runs in seconds).

**1. Symptom frequency / class balance — hours, not days.**

Can be computed immediately from data already on disk, no rerun needed: either threshold the
existing per-checkpoint columns in `results/classifier_seedfix/causal_feature_matrix.csv`
(`separated_after_contact_now`, `goal_offset_now`, `safety_violation_rate_sofar`, etc.) or
derive per-episode boolean flags by lightly modifying `label_episode()`'s existing internal
checks (`_detect_grasp`, `_detect_drop`, goal-spoof condition, `_is_diverging`) to return all
matches instead of the first. Either path is direct reuse of numbers/logic that already exist
— estimate **2-4 hours**, mostly for the second (exact) path since the first (approximate) path
is a one-line `.value_counts()`-equivalent pass over an already-loaded CSV.

**Rarity calibration, computed directly from the existing single-label data** (all 4 models,
6,600 episodes, `results/data_seedfix/sac_her_only/labels_sac_her_pickandplace_all.csv`):

| label | n | % |
|---|---|---|
| never_reached_object | 4,440 | 67.3% |
| reached_but_failed_grasp | 1,012 | 15.3% |
| success | 549 | 8.3% |
| spoofed_goal | 295 | 4.5% |
| divergent_transport | 178 | 2.7% |
| grasped_but_dropped | 126 | 1.9% |

This is directly relevant, not just background: the rarest existing single label
(`grasped_but_dropped`, 1.9% of episodes) is already flagged in findings.md's Open Issues §2
as having only **2 test examples in the seed-4 split** — "performance estimate is not
meaningful." Multi-label symptoms are, by construction, decompositions of these same
episodes into finer-grained flags (e.g. a drop symptom, a goal-conflict symptom, a
transport-divergence symptom, computed independently rather than collapsed to one label per
episode) — so several of the 8 symptom types should be expected to land at or below the
1.9-2.7% range, i.e. in the same "too few test examples to be meaningful" territory that
`grasped_but_dropped` already hit as a single label. This isn't a new risk uncovered by this
scoping pass — it's the same problem the Level 1 false-alarm work hit (n=30/checkpoint was too
sparse to cite directly, forcing the early/mid/late binning workaround) — but it will likely
hit *harder* for multi-label, since the whole point of decomposing into symptoms is finer
granularity than the six-label taxonomy, which means smaller per-symptom n by construction.

**2. Per-symptom validation scope — the dominant cost, ~1.5-2 days total across all 8.**

Important scoping note: there is no independent human-annotated ground truth for either the
existing six-label taxonomy or any multi-label extension of it — "precision/recall" here means
*classifier-predicted symptom vs. rule-derived symptom flag*, the same relationship
`train_causal_classifier.py`/`bootstrap_false_alarm_ci.py` already have to the rule-based
`label_episode()` output. That keeps this in the same methodological family as the Level 1
work (not a new validation paradigm), but it is not free:

- Precision/recall/false-alarm-rate for a single-label multiclass problem (the existing
  6-label case) is one confusion matrix. For 8 independent binary symptoms, either (a) train 8
  binary classifiers, or (b) restructure the existing pooled multiclass RF's evaluation loop
  into 8 one-vs-rest passes. Either way this is a real adaptation of
  `train_causal_classifier.py`'s evaluation code, not a parameter change — estimate **~1 day**
  for the adaptation + first-pass run across all 8 symptoms (the RF architecture, feature set,
  and train/test seed split are all reusable as-is; what's being rebuilt is the label/metric
  plumbing).
- Bootstrap CIs, following the exact pattern of `bootstrap_false_alarm_ci.py` (load saved
  model, no retrain, 5,000 percentile resamples), generalized to loop over 8 symptoms instead
  of one false-alarm-rate subset — mechanically closer to "half a day" than "half a day × 8"
  since the resampling code itself doesn't change per symptom. But per the rarity finding
  above, an unknown-until-computed subset of the 8 symptoms will need the same early/mid/late
  (or equivalent) binning workaround the false-alarm work needed, and that binning strategy
  isn't mechanical — it was a judgment call about which checkpoints to pool, done once for one
  metric on 2026-07-20. Doing that judgment call per-rare-symptom, potentially 3-5 times, is
  the actual long pole here. Estimate **~half a day to 1 day**, wide range because it's
  contingent on how many symptoms turn out rare.
- **Total for item 2: ~1.5-2 days**, with the caveat that this assumes the rarest symptoms
  don't need a bespoke fix beyond binning (e.g. pooling across models/conditions) — if any
  symptom is rare enough that even binned CIs are uninformative (plausible given
  `grasped_but_dropped`'s 2-example precedent), the honest outcome for that symptom is "cannot
  be validated to CI-citable rigor with current data," not more engineering time fixing it.

**3. Label plausibility / correlation sanity check — half a day, one-time.**

Smaller and bounded: a co-occurrence crosstab of derived symptom flags (does "dropped" +
"goal-conflict" occur in plausible conditions, e.g. concentrated under `goal_spoof_*` and
`object_pose_spoof` rather than uniformly everywhere) plus spot-checking a handful of
individual episode trajectories against the flagged combination, the same kind of manual/
semi-manual check `scripts/phase7_spotcheck.py` already established as a precedent at this
project's scale. This is a one-time check, not per-symptom, and reuses the same
crosstab-over-existing-CSV technique used earlier in this investigation for Q2. Estimate
**2-4 hours**.

**4. Total, end-to-end, for paper-citable multi-label Level 3:**

| step | estimate |
|---|---|
| Label derivation (exact, per prior investigation) | ~hours (few hours – 1 day, incl. the optional step_logs rerun if exact fidelity is required) |
| 1. Symptom frequency / class balance | 2-4 hours |
| 2. Per-symptom precision/recall/false-alarm + bootstrap CIs (8 symptoms) | 1.5-2 days |
| 3. Plausibility / co-occurrence sanity check | 2-4 hours (half a day) |
| **Total** | **~2.5-3.5 days** of focused work, dominated by item 2 |

This is a rough order-of-magnitude, not a committed schedule — actual time is contingent on
how many of the 8 symptoms land in the sub-2% rarity range found for `grasped_but_dropped`,
since those will need the same binning judgment calls the Level 1 false-alarm work needed, and
some may not be citable at CI-level rigor at all regardless of time spent (a data-limit, not an
engineering-time problem). The single-label "reuse existing taxonomy" path has no equivalent
per-symptom validation burden — its precision/recall/false-alarm numbers are already computed
and citable from the Phase 8/8.5/9 work already in `findings.md`.

---

## Files touched

- Created/appended: this report only (`results/investigation/level3_multilabel_and_taxonomy_findings.md`).
  The validation-cost-scoping section was appended to the existing findings file rather than
  a new file, since it directly extends Q1's derivation-cost estimate — kept the two together
  so the full derivation+validation cost picture reads as one document.
- No existing labels, data files, or training/scoring scripts were modified.
- No multi-label data was derived or generated; no validation step was executed; no
  `step_logs_*.csv` rerun was performed.
- No code was run except read-only inspection: `pandas`/`csv` crosstabs over already-existing
  CSVs (`labels_sac_her_pickandplace_all.csv`), `git show`/`git log` to inspect the
  `retrain/causal-flat-rf-baseline` branch's bootstrap-CI script and findings.md without
  checking that branch out. No simulator, no training, no model calls.
- `retrain/causal-flat-rf-baseline` and Phase 1 work were not touched or checked out — this
  work happened entirely on `investigate/level3-multilabel-and-taxonomy`, branched from `main`.
