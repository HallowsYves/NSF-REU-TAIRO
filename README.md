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
decisions conditioned on a trained failure-mode classifier.

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

Recovery v4 (CCAR) is implemented and evaluated on the `clean_2M` PickAndPlace checkpoint,
replacing the threshold-based fallback the original paper draft described as still in
development. An online/causal failure-mode classifier (six-label taxonomy, Phase 9 complete)
is trained and feeds the recovery controller. Phase B/C dense-classifier sweep figures and
episode results are committed. The dense-feature extension is analytically complete; Tier 2
(broader checkpoint coverage) remains documented future work in `RECOVERY_V4.md`.

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
│                                #   v4 (CCAR — Classifier-Conditioned Adaptive Recovery)
├── scripts/                     # Sweep entrypoint, figure builders, training scripts,
│                                #   diagnostic/calibration scripts (see findings.md for history)
├── training/                    # SAC+HER training, attack-aware and single-attack wrappers
├── notebooks/                   # Supporting Jupyter notebooks
├── results/                     # Local only — gitignored (see below)
│   ├── models/                  # Trained checkpoints + replay buffers
│   ├── data_seedfix/            # Canonical seed-fixed episode results (authoritative numbers)
│   ├── data_recovery_v4/        # Recovery v4 Phase 6 evaluation (episode_results committed)
│   ├── data_recovery_v4_dense/  # Dense-sweep results (episode_results committed)
│   ├── classifier/              # Original pre-fix RF classifier (historical)
│   ├── classifier_seedfix/      # Seed-fixed RF + causal + online + v4 calibration
│   ├── classifier_seedfix_dense/# Dense-feature variant (1.4 GB, local only)
│   ├── figures/                 # Diagnostic and publication plots
│   └── archive/                 # Pre-fix data (README committed; CSVs gitignored)
├── RECOVERY_V4.md               # CCAR design and implementation status
├── important_update_for_paper.md# Pre-submission paper update checklist
├── findings.md                  # Phase-by-phase failure-mode classifier findings (Phases 0–9)
├── ATTACK_AWARE_TRACK.md        # Attack-aware policy track (ground-truth flag, Dr. Ho proposal)
└── update_paper.md              # Paper-writing guide with verified numbers and source citations
```

The `results/` tree is fully gitignored except for `results/data_recovery_v4/` episode CSVs,
`results/data_recovery_v4_dense/` episode CSVs, and `results/archive/README.md`. All large
artifacts (models, data CSVs, classifier pickles) must be obtained from the source TAIRO repo
or regenerated locally.

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

# Specific conditions and model
python3 scripts/run_multiseed_sweep.py \
    --conditions sensor_dropout sensor_bias goal_spoof_midep \
    --model-path results/models/sac_her_pickandplace_clean_2M
```

Results land in `results/data_seedfix/` by default. Summary tables are built via
`scripts/build_benchmark_table.py`.

---

## Paper

The current LaTeX draft is `paper/NSF_REU_2026_TAIRO_WEEK_8/TAIRO_PAPER.tex` (Draft 3).
Before finalizing, check `important_update_for_paper.md` for the prioritized list of
corrections and additions needed, and `update_paper.md` for verified numbers with source
citations.

---

## Experimental History

Full phase-by-phase record of the failure-mode classifier workstream (Phases 0–9) lives in
`findings.md`. Recovery v4 design, implementation, and evaluation details are in
`RECOVERY_V4.md`. Attack-aware policy track is in `ATTACK_AWARE_TRACK.md`.
