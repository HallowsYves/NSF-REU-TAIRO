# TAIRO — Trustworthiness Evaluation of Robotic Policies Under Adversarial Cyberattacks

A recovery-aware evaluation framework that benchmarks how cyber-physical attacks degrade
robotic manipulation policies, and uses failure detection together with recovery-aware control
to decide when and how a robot should recover.

---

## Motivation

Existing adversarial-robustness benchmarks for reinforcement learning mostly measure how
much task success degrades under a perturbation, but they generally don't address whether a
system can detect and recover from attack-induced failure online, whether robustness should
be scored through a broader trustworthiness lens rather than success rate alone, or how the
relationship between attack surface and policy architecture shapes failure modes. TAIRO
targets that gap for robotic manipulation specifically, where sparse rewards, contact-rich
dynamics, and closed-loop control make failures abrupt and hard to diagnose.

---

## Approach

**Policy and environment.** A Soft Actor-Critic policy with Hindsight Experience Replay
(SAC+HER), evaluated on the contact-rich, sparse-reward FetchPickAndPlace-v4 manipulation
task (Fetch manipulator grasping a free-floating object and moving it to a target position),
building on an earlier validation pilot on the lower-dimensional FetchReach-v4 task.

**Attack taxonomy.** An eleven-condition cybersecurity attack taxonomy spanning observation-,
goal-, and action-space perturbations, including three attacks specific to object manipulation:
object pose spoofing, gripper-state falsification, and contact dropout.

**Recovery system.** An iteratively designed recovery system that detects attack-like behavior
online and intervenes accordingly — a threshold-based fallback controller (v2/v3) validated
on the FetchReach-v4 pilot, and a classifier-conditioned adaptive recovery controller
(CCAR / Recovery v4) for PickAndPlace, replacing the fixed-threshold approach with recovery
decisions conditioned on a trained failure-mode classifier. A five-level hierarchical
failure-diagnosis stack (TAIRO-HX: task stage → anomaly detection → failure type → attack
family → recovery decision) feeds two of those levels back into the controller itself —
`sac_her_recovery_v4_hx2` (Level 1 stage-gate + Level 4 attack-family down-weight) is the
adopted final controller, fixing a real regression in plain Recovery v4 without giving up
its safety profile. See `RECOVERY_V4.md` and `CLAUDE.md`'s Level-Chaining Architecture
section for the full design.

**Trustworthiness scoring.** A five-component composite trustworthiness score (C1–C5)
grounded in the NIST AI Risk Management Framework, scoring reliability, robustness, cyber
resilience, safety, and recovery rather than raw success rate alone.

---

## Key Findings

- **Vulnerability is strongly attack-surface dependent.** SAC+HER holds up under several
  action-space attacks (clipping, delay — full success) but collapses under sensor and
  goal-channel corruption (sensor dropout, sensor bias, action reversal, gripper falsification
  — near-total failure) and is substantially degraded under goal spoofing and object-pose
  spoofing.

- **A scoped negative result.** Domain-randomized training failed to learn the grasp phase at
  either 500k or 2M training steps, showing that robustness-oriented training strategies need
  enough clean-task signal left to actually acquire the base skill they are meant to protect.

- **Recovery effectiveness depends on detector-attack alignment.** A detector tuned for one
  channel can be structurally blind to damage on another. Trustworthiness scoring is
  outcome-referenced — recovery is credited only when it improves task success, not merely
  for triggering frequently.

---

## Current Status

*(Last synced 2026-07-22.)*

Recovery v4 (CCAR) is implemented and evaluated on the `clean_2M` PickAndPlace checkpoint,
replacing the threshold-based fallback the original paper draft described as still in
development. An online/causal failure-mode classifier (six-label taxonomy, Phase 9 complete)
is trained and feeds the recovery controller. Phase B/C dense-classifier sweep figures and
episode results are committed. The dense-feature extension is analytically complete; Tier 2
(broader checkpoint coverage) remains documented future work in `RECOVERY_V4.md`.

**TAIRO-HX hierarchy (Levels 1–5) is complete**, and two levels are wired back into the
recovery controller itself (not just an offline analysis track): `sac_her_recovery_v4_hx2`
(Level 1 stage-gate + Level 4 attack-family down-weight) is the **adopted final recovery
method**, confirmed at full statistical power (n=450, seeds 0–14, all 11 conditions) to fix
a real, previously undocumented harm in plain Recovery v4 on `grip_state_falsification`
while preserving v4's much lower safety-violation rate. Four other variants
(`v4_hx`, `v4_hx3`, `v4_hx4`, `v4_hx5`, `v4_hx6`) were built and evaluated at the same power
as part of this investigation; none were adopted — see `RECOVERY_V4.md` for the full,
documented negative-result history (this matters for reproducing the process, not just the
final numbers).

**Final mentor-requested comparison** (no recovery vs. earlier baselines v2/v3 vs.
gradual-response v4 vs. final selective v4-HX2, all 8 requested metrics: task-success rate,
clean-task performance, detection delay, recovery-response delay, recovery time, safety
violations, number of interventions, completion-time overhead) is in
`results/recovery_hx_results_summary.md` §6, with figures in
`results/figures/final_hx_comparison/`. Headline: it is **not** a clean "v4-HX2 wins
everywhere" story — the older v2/v3 baselines still significantly outperform v4-HX2 on
`grip_state_falsification` itself (the one condition with a confirmed effect), because their
hard-override architecture responds in ~9 steps vs. v4-HX2's ~90+ step gradual ramp; v4-HX2's
advantage is a much lower safety-violation rate. Both properties are real and documented,
not glossed over.

**A live interactive demo** (`app/live_attack_demo.py`, Streamlit) runs the raw SAC+HER
policy side-by-side with SAC+HER + Recovery v4-HX2 against any of the 11 attack conditions,
with live classifier/recovery telemetry and an end-of-episode trustworthiness comparison
against the committed benchmark. See "Running the Live Demo" below.

---

## Team and Acknowledgments

Yves Velasquez Vega (California State University, Fullerton), Jachin Choi (Case Western
Reserve University), Sunny Sood and Abhinav Kochar (University of Missouri-Kansas City),
advised by Dr. Duy Ho (CSU Fullerton) and Dr. Yugyung Lee (UMKC).

NSF REU Site "AI-empowered Cybersecurity," Grant CNS-2349236.

---

## Repository Structure

```
NSF-REU-TAIRO/
├── config.py                    # Single source of truth for all constants
├── requirements.txt
├── attacks/                     # Attack implementations (action + sensor channels)
├── app/                         # Live interactive Streamlit demo (recovery-comparison)
│   ├── live_attack_demo.py      #   UI/session-state/control flow
│   └── sim_worker.py            #   subprocess-isolated MuJoCo rendering (macOS-required)
├── envs/                        # FetchReach and FetchPickAndPlace env wrappers
├── evaluation/                  # Episode runner, attack dispatch, metrics, labeling,
│                                #   causal feature builder (Phase 9)
├── paper/
│   └── NSF_REU_2026_TAIRO_WEEK_8/
│       ├── TAIRO_PAPER.tex      # Draft 3 (current working paper)
│       ├── TAIRO_PAPER.cls
│       ├── references.bib
│       └── paper_figures/       # 9 committed figures (5 main + 3 Recovery v4 + 1 framework)
├── policies/                    # Rule-based and SAC+HER policy wrappers
├── recovery/                    # v2 (step-at-a-time), v3 (sustained window),
│                                #   v4 (CCAR), v4_hx..v4_hx6 (TAIRO-HX variants —
│                                #   v4_hx2 is the sole adopted variant; see RECOVERY_V4.md)
├── scripts/                     # Sweep entrypoint, figure builders, training scripts,
│                                #   diagnostic/calibration scripts (see findings.md for history)
│                                #   build_final_hx_comparison.py / build_final_hx_figures.py —
│                                #   the final mentor-requested 4-arm comparison
├── training/                    # SAC+HER training, attack-aware and single-attack wrappers
├── notebooks/                   # Supporting Jupyter notebooks
├── results/                     # Local only — gitignored (see below)
│   ├── models/                  # Trained checkpoints + replay buffers
│   ├── data_seedfix/            # Canonical seed-fixed episode results (authoritative numbers)
│   ├── data_recovery_v4/        # Recovery v4 Phase 6 evaluation (episode_results committed)
│   ├── data_recovery_v4_dense/  # Dense-sweep results (episode_results committed)
│   ├── data_recovery_v4_hx*/    # v4_hx..v4_hx6 evaluation (episode_results committed)
│   ├── data_recovery_v4_v2v3_backfill/  # v2/v3 seeds-5-14 backfill (episode_results committed)
│   ├── data_recovery_v4_power_check*/   # seeds-5-14 backfill for sac_her/v4/v4_hx/v4_hx2
│   ├── classifier/              # Original pre-fix RF classifier (historical)
│   ├── classifier_seedfix/      # Seed-fixed RF + causal + online + v4 calibration
│   ├── classifier_seedfix_dense/# Dense-feature variant (1.4 GB, local only)
│   ├── classifier_level4/       # Level 4 (attack-family) classifier
│   ├── figures/                 # Diagnostic and publication plots, incl.
│   │                            #   figures/final_hx_comparison/ (final 4-arm figures)
│   └── archive/                 # Pre-fix data (README committed; CSVs gitignored)
├── RECOVERY_V4.md               # CCAR + TAIRO-HX design and implementation status
├── important_update_for_paper.md# Pre-submission paper update checklist
├── findings.md                  # Phase-by-phase failure-mode classifier findings (Phases 0–12)
├── ATTACK_AWARE_TRACK.md        # Attack-aware policy track (ground-truth flag)
└── update_paper.md              # Paper-writing guide with verified numbers and source citations
```

The `results/` tree is fully gitignored except for episode-result CSVs (not the much larger
per-step logs) across `results/data_recovery_v4*/` — this includes the Phase 6 evaluation,
the dense sweep, every TAIRO-HX variant's evaluation, the v2/v3 backfill, and the seeds-5-14
power-check directories — plus `results/archive/README.md` and small metrics/summary CSVs
under `results/classifier_level1/`, `results/classifier_level4/`, `results/final_hx_comparison_*`.
All large artifacts (models, per-step-log CSVs, classifier pickles) must be obtained from the
source TAIRO repo or regenerated locally.

---

## Environment Setup

```bash
# Activate the project conda environment
conda activate reu_robotics

# Verify key packages (Python 3.11)
python3 -c "import gymnasium, stable_baselines3, sklearn; print('OK')"
```

See `requirements.txt` for the full dependency list. The environment was built against
`stable_baselines3==2.8.0` and `gymnasium-robotics` for the Fetch environment suite.

---

## Running a Benchmark Sweep

```bash
# Default sweep: sac_her + v2 + v3, all 11 conditions, 5 seeds × 30 eps
python3 scripts/run_multiseed_sweep.py

# Include Recovery v4 (requires classifier artifacts in results/classifier_seedfix/)
python3 scripts/run_multiseed_sweep.py --methods sac_her sac_her_recovery_v4

# Include the adopted TAIRO-HX final controller (requires both
# results/classifier_seedfix/ AND results/classifier_level4/level4_classifier.pkl)
python3 scripts/run_multiseed_sweep.py \
    --env pickandplace --model-path results/models/sac_her_pickandplace_clean_2M \
    --methods sac_her_recovery_v4_hx2 \
    --recovery-v4-classifier-dir results/classifier_seedfix

# Specific conditions and model
python3 scripts/run_multiseed_sweep.py \
    --conditions sensor_dropout sensor_bias goal_spoof_midep \
    --model-path results/models/sac_her_pickandplace_clean_2M
```

Results land in `results/data_seedfix/` by default. Summary tables are built via
`scripts/build_benchmark_table.py`. The final mentor-requested 4-arm comparison (no
recovery / v2+v3 / v4 / v4-HX2, all 8 requested metrics) is reproduced via:

```bash
python3 scripts/build_final_hx_comparison.py   # statistics -> results/final_hx_comparison_*.csv
python3 scripts/build_final_hx_figures.py      # tables + figures -> results/figures/final_hx_comparison/
```

---

## Running the Live Demo

```bash
conda activate reu_robotics
streamlit run app/live_attack_demo.py
```

Runs the raw SAC+HER `clean_2M` PickAndPlace policy side-by-side with SAC+HER +
Recovery v4-HX2 (the adopted final controller) against any of the 11 attack conditions,
with a live classifier/recovery telemetry panel and an end-of-episode trustworthiness
comparison against the committed benchmark. Requires
`results/classifier_seedfix/{online_failure_classifier.pkl,recovery_v4_trigger_calibration.pkl}`
and `results/classifier_level4/level4_classifier.pkl`. Try condition
`grip_state_falsification` with seed `114` for a verified example of recovery saving an
otherwise-failed episode (see `app/live_attack_demo.py`'s `SUGGESTED_EXAMPLE`).

---

## Paper

The current LaTeX draft is `paper/NSF_REU_2026_TAIRO_WEEK_8/TAIRO_PAPER.tex` (Draft 3).
Before finalizing, check `important_update_for_paper.md` for the prioritized list of
corrections and additions needed, and `update_paper.md` for verified numbers with source
citations.

---

## Experimental History

Full phase-by-phase record of the failure-mode classifier workstream (Phases 0–12,
including the TAIRO-HX hierarchy build and the recovery-integration/final-comparison work)
lives in `findings.md`. Recovery v4 + TAIRO-HX design, implementation, and evaluation
details are in `RECOVERY_V4.md`. Attack-aware policy track is in `ATTACK_AWARE_TRACK.md`.
`results/recovery_hx_results_summary.md` is the standalone results package (tables, charts,
statistical comparisons, interpretations) built for the poster/video deliverable — start
there for a narrative walkthrough rather than the raw phase-by-phase logs.
