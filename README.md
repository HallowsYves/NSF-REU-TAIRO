# TAIRO — Trustworthiness Evaluation of Robotic Policies Under Adversarial Cyberattacks

A recovery-aware evaluation framework that benchmarks how cyber-physical attacks degrade
robotic manipulation policies, and uses failure detection together with recovery-aware control
to decide when and how a robot should recover.

*Last synced 2026-07-23. Adopted final controller: `sac_her_recovery_v4_hx6` (see
[Current Status](#current-status) and `FINAL_APPROACH.md`).*

---

## Contents

- [Motivation](#motivation)
- [Approach](#approach)
- [Key Findings](#key-findings)
- [Current Status](#current-status)
- [Documentation Map](#documentation-map)
- [Team and Acknowledgments](#team-and-acknowledgments)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Running a Benchmark Sweep](#running-a-benchmark-sweep)
- [Running the Live Demo](#running-the-live-demo)
- [Paper](#paper)
- [Experimental History](#experimental-history)

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

**Recovery system — three generations.**

| Generation | Mechanism | Status |
|---|---|---|
| **v2 / v3** | Hard-threshold trigger: rule-based signals fire a full, unblended override for a sustained window once tripped. | Earlier baselines |
| **v4 (CCAR)** | Classifier-Conditioned Adaptive Recovery: a trained failure-mode classifier drives a continuous blend weight between the policy's action and one of five expert recovery controllers. | Superseded |
| **v4-HX2 → v4-HX6 (adopted)** | v4's continuous blend, plus TAIRO-HX's hierarchical diagnosis (task stage → anomaly → failure type → attack family → recovery decision) fed back into the controller: a Level-1 stage-gate, a Level-4 attack-family down-weight, and (new in HX6) a Level-4-gated fast-attack trigger. | **Final** |

`sac_her_recovery_v4_hx6` is the adopted final controller: it fixes a confirmed regression in
plain Recovery v4 on `grip_state_falsification` (same fix as HX2), and additionally speeds up
detection on perception/goal-family attacks without reopening that fix. See `FINAL_APPROACH.md`
for the full, standalone writeup of what changed and how it compares to every baseline, and
`RECOVERY_V4.md` plus `CLAUDE.md`'s Level-Chaining Architecture section for implementation detail.

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

- **Speed and safety trade off across recovery architectures, not just within one.** On the one
  condition with a confirmed recovery effect (`grip_state_falsification`), the older hard-override
  baselines (v2/v3) still beat the final controller outright on raw success — because they react
  in ~9 steps vs. v4-HX6's gradual ~87–92-step ramp — but at 30–65× the safety-violation rate.
  Neither property is hidden in favor of the other; see `FINAL_APPROACH.md` §3.

---

## Current Status

*(Last synced 2026-07-23.)*

**`sac_her_recovery_v4_hx6` is the adopted final recovery controller**, wired into
`evaluation/episode_runner.py`, `scripts/run_multiseed_sweep.py`, and the live Streamlit demo.
It layers two TAIRO-HX-derived refinements onto Recovery v4 (CCAR):

1. A **Level-4 attack-family down-weight** (inherited from v4-HX2) that reduces the recovery
   expert's blend weight when an `action_actuation`-family attack is confidently detected —
   fixing a confirmed regression where plain v4 did statistically significant harm on
   `grip_state_falsification` relative to no recovery at all (BH-adjusted p=0.0002).
2. A **Level-4-gated fast-attack trigger** (new in HX6) that speeds up detection specifically
   on `perception_state`/`goal_manipulation`-family attacks, without touching the trigger speed
   — and therefore without risk to the Addition-1 fix — on `action_actuation`-family attacks.

Three intermediate variants (`v4_hx3`, `v4_hx4`, `v4_hx5`) targeted the goal-spoofing latency
gap directly and were **confirmed nulls**; v4-HX6 was adopted anyway because it is a strict,
zero-regression improvement over v4-HX2 rather than a new confirmed win — flagged explicitly
in `FINAL_APPROACH.md` as a different adoption bar than every prior variant in this project.

**Final mentor-requested comparison** (no recovery vs. v2/v3 vs. gradual-response v4 vs. final
v4-HX6, all 8 requested metrics: task-success rate, clean-task performance, detection delay,
recovery-response delay, recovery time, safety violations, number of interventions,
completion-time overhead) is in `results/recovery_hx_results_summary.md` §6 and
`FINAL_APPROACH.md` §4, with figures in `results/figures/final_hx_comparison/`. It is **not** a
clean "v4-HX6 wins everywhere" story: v2/v3 still significantly outperform v4-HX6 on
`grip_state_falsification` itself — the one condition with a confirmed effect — because of the
latency gap above; v4-HX6's advantage is a safety-violation rate roughly 1/30th to 1/65th theirs.

**A live interactive demo** (`app/live_attack_demo.py`, Streamlit) runs the raw SAC+HER policy
side-by-side with SAC+HER + Recovery v4-HX6 against any of the 11 attack conditions, with live
classifier/recovery telemetry and an end-of-episode trustworthiness comparison against the
committed benchmark. See [Running the Live Demo](#running-the-live-demo) below.

Also complete: the TAIRO-HX hierarchy (Levels 1–5), the online/causal failure-mode classifier
(six-label taxonomy, Phase 9), and the Phase B/C dense-classifier sweep. The dense-feature
extension is analytically complete; Tier 2 (broader checkpoint coverage) remains documented
future work in `RECOVERY_V4.md`.

---

## Documentation Map

The repo has several standalone `.md` docs at different levels of depth. Start with whichever
matches what you need:

| Doc | What it's for |
|---|---|
| `FINAL_APPROACH.md` | **Start here for the current approach.** Standalone explanation of what v4-HX6 is, why each piece exists, and exactly how it compares to every baseline — doesn't assume you've read the others first. |
| `POSTER_DRAFT.md` | Panel-by-panel draft copy for the IEEE BigData 2026 REU Symposium poster; a condensed, presentation-ready version of `FINAL_APPROACH.md`. |
| `RECOVERY_V4.md` | Full CCAR + TAIRO-HX design and implementation status, including the complete negative-result history behind `v4_hx3`–`v4_hx5`. |
| `findings.md` | Phase-by-phase record of the failure-mode classifier workstream (Phases 0–12). |
| `TAIRO-HX.md` | The five-level hierarchical failure-diagnosis stack itself (task stage → anomaly → failure type → attack family → recovery decision). |
| `results/recovery_hx_results_summary.md` | Standalone results package (tables, charts, statistical comparisons) built for the poster/video deliverable — best narrative walkthrough of the numbers. |
| `ATTACK_AWARE_TRACK.md` | The separate attack-aware policy track (ground-truth attack flag as input). |
| `reported_grasp_contact_options.md` | Reference notes on grasp/contact-detection options considered for the manipulation task. |
| `update_paper.md` | Verified numbers with source citations, for writing the paper. |
| `important_update_for_paper.md` | Prioritized checklist of corrections/additions needed before the paper is finalized. |

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
│                                #   v4_hx6 is the adopted variant; see RECOVERY_V4.md)
├── scripts/                     # Sweep entrypoint, figure builders, training scripts,
│                                #   diagnostic/calibration scripts (see findings.md for history)
│                                #   build_final_hx_comparison.py / build_final_hx_figures.py —
│                                #   the final mentor-requested 4-arm comparison
├── training/                    # SAC+HER training, attack-aware and single-attack wrappers
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
├── FINAL_APPROACH.md            # Standalone writeup: final approach vs. every baseline
├── POSTER_DRAFT.md              # IEEE BigData 2026 REU Symposium poster draft copy
├── TAIRO-HX.md                  # The five-level hierarchical failure-diagnosis stack
├── reported_grasp_contact_options.md # Grasp/contact-detection design notes
├── important_update_for_paper.md# Pre-submission paper update checklist
├── findings.md                  # Phase-by-phase failure-mode classifier findings (Phases 0–12)
├── ATTACK_AWARE_TRACK.md        # Attack-aware policy track (ground-truth flag)
└── update_paper.md              # Paper-writing guide with verified numbers and source citations
```

> **Note:** an earlier version of this structure listed a top-level `notebooks/` directory that
> no longer appears in the repo file listing — worth confirming whether it was removed
> intentionally or should be restored/gitignored explicitly.

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
    --methods sac_her_recovery_v4_hx6 \
    --recovery-v4-classifier-dir results/classifier_seedfix

# Specific conditions and model
python3 scripts/run_multiseed_sweep.py \
    --conditions sensor_dropout sensor_bias goal_spoof_midep \
    --model-path results/models/sac_her_pickandplace_clean_2M
```

Results land in `results/data_seedfix/` by default. Summary tables are built via
`scripts/build_benchmark_table.py`. The final mentor-requested 4-arm comparison (no
recovery / v2+v3 / v4 / v4-HX6, all 8 requested metrics) is reproduced via:

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
Recovery v4-HX6 (the adopted final controller) against any of the 11 attack conditions,
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
details are in `RECOVERY_V4.md`; the final approach and its comparison against every baseline
is in `FINAL_APPROACH.md`. Attack-aware policy track is in `ATTACK_AWARE_TRACK.md`.
`results/recovery_hx_results_summary.md` is the standalone results package (tables, charts,
statistical comparisons, interpretations) built for the poster/video deliverable — start
there for a narrative walkthrough rather than the raw phase-by-phase logs.
