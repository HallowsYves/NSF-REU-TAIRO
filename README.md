# TAIRO — Trustworthiness Evaluation of Robotic Policies Under Adversarial Cyberattacks

A recovery-aware evaluation framework that benchmarks how cyber-physical attacks degrade
robotic manipulation policies, and uses failure detection together with recovery-aware control
to decide when and how a robot should recover.

*Last synced 2026-07-29. Adopted final controller: `sac_her_recovery_v4_hx6` (see
[Current Status](#current-status)).*

---

## Contents

- [Motivation](#motivation)
- [Approach](#approach)
- [Key Findings](#key-findings)
- [Current Status](#current-status)
- [Recovery Variant Comparison: HX Through HX6](#recovery-variant-comparison-hx-through-hx6)
- [Documentation Map](#documentation-map)
- [Team and Acknowledgments](#team-and-acknowledgments)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Running a Benchmark Sweep](#running-a-benchmark-sweep)
- [Running the Live Demo](#running-the-live-demo)
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
detection on perception/goal-family attacks without reopening that fix. See
[Current Status](#current-status) below for how it compares to every baseline.

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
  Neither property is hidden in favor of the other; see [Current Status](#current-status) below.

---

## Current Status

*(Last synced 2026-07-29.)*

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
zero-regression improvement over v4-HX2 rather than a new confirmed win — a different adoption
bar than every prior variant in this project.

**Final mentor-requested comparison** (no recovery vs. v2/v3 vs. gradual-response v4 vs. final
v4-HX6, all 8 requested metrics: task-success rate, clean-task performance, detection delay,
recovery-response delay, recovery time, safety violations, number of interventions,
completion-time overhead), with figures in `results/figures/final_hx_comparison/`. It is
**not** a clean "v4-HX6 wins everywhere" story: v2/v3 still significantly outperform v4-HX6 on
`grip_state_falsification` itself — the one condition with a confirmed effect — because of the
latency gap above; v4-HX6's advantage is a safety-violation rate roughly 1/30th to 1/65th theirs.

**A live interactive demo** (`app/live_attack_demo.py`, Streamlit) runs the raw SAC+HER policy
side-by-side with SAC+HER + Recovery v4-HX6 against any of the 11 attack conditions, with live
classifier/recovery telemetry and an end-of-episode trustworthiness comparison against the
committed benchmark. See [Running the Live Demo](#running-the-live-demo) below.

Also complete: the TAIRO-HX hierarchy (Levels 1–5), the online/causal failure-mode classifier
(six-label taxonomy, Phase 9), and the Phase B/C dense-classifier sweep. The dense-feature
extension is analytically complete; Tier 2 (broader checkpoint coverage) remains future work.

---

## Recovery Variant Comparison: HX Through HX6

*Last updated 2026-07-23. All numbers are n=450 (seeds 0–14, all 11 attack conditions,
clean_2M PickAndPlace checkpoint) unless otherwise noted. Benjamini-Hochberg FDR-corrected
for multi-variant comparisons.*

### Baseline: Recovery v4 (CCAR)

**File:** `recovery/recovery_v4.py`

Classifier-Conditioned Adaptive Recovery — the starting point for all HX variants.

**Core formula:**
```python
final_action = (1-w) * policy_action + w * recovery_action
```

where `w` is an EMA-smoothed blend weight derived from the online failure-mode classifier's
`p_fail` output, and `recovery_action` is a weighted mixture of five hand-built expert
controllers.

**Five experts:** `transport_expert`, `relocalization_expert`, `grasp_stabilize_expert`,
`regrasp_expert`, `approach_expert`.

**Trigger mechanism:** `TriggerWeight` class in `recovery_v4.py`. When `p_fail` rises, alpha
increases (faster blend); when it falls, alpha decreases (slower retreat). The weight crosses
activation thresholds gradually, not instantly.

**Known defect (from `recovery_do_no_harm_audit.csv`):** Plain v4 does statistically
significant **harm** on `grip_state_falsification` relative to doing nothing: 19.8% → 14.0%
success (BH p=0.00017). This is the motivating problem that HX2 was built to fix.

### HX (v1) — Level 1 Stage-Gating

**File:** `recovery/recovery_v4_hx.py` (~126 lines)

**Motivation.** The five expert controllers in v4 are designed for different task phases
(approach, grasp, transport, etc.). Without stage-gating, a transport expert could fire during
the approach phase and steer the arm toward the goal prematurely, or a grasp expert could fire
after the object is already transported and push the arm back.

**Implementation.** New function `compute_task_stage_online()` — a 5-feature priority cascade
computed from `feature_vec`:

```
is_success_now → verifying_completion
grasp_kinematic_ever_sofar → transporting
dttg_now < LEVEL1_PLACE_RADIUS → placing
contact_streak_now >= 1 → grasping
reached_ever_sofar → aligning_gripper
otherwise → approaching_object
```

Modified `compute_recovery_action_hx()` — each expert's weight is multiplied by a
`stage_factor`: `1.0` if the current `task_stage` is in the expert's compatible stage set,
`config.STAGE_EXPERT_SOFT_WEIGHT` (0.15) otherwise (soft, not hard-zero). Stage-expert
compatibility mapping from `config.STAGE_EXPERT_COMPAT` — e.g., `approach_expert` is
compatible with `approaching_object` and `aligning_gripper`; `transport_expert` is compatible
with `transporting` and `placing`.

**Results.**

| Condition | v4 success | HX success | Delta | BH-adjusted p |
|---|---|---|---|---|
| `object_pose_spoof` | 28.2% | 21.3% | -6.9pp | 0.180 |
| All other conditions | — | — | ~0 | ~1.0 |

**No confirmed benefit on any condition.** The `object_pose_spoof` regression (-6.9pp) is not
significant after full-grid BH correction (p=0.180). Stage-gating does not improve recovery
performance and was not adopted.

### HX2 (v2) — + Level 4 Attack-Family Down-Weight

**File:** `recovery/recovery_v4_hx2.py` (~95 lines)

**Motivation.** Plain v4 does significant harm on `grip_state_falsification` (19.8% → 14.0%).
Root cause: `grip_state_falsification` is an `action_actuation`-family attack — it corrupts
the command channel itself. A recovery expert that reasons about *state* (object position,
gripper aperture) has no leverage there; blending its output adds noise on an already-corrupted
actuation path.

**Implementation.** Adds Level 4 (attack-family) classifier query **after**
`compute_recovery_action_hx()` returns:

```python
l4_pred_class = level4_classifier["label_encoder"].inverse_transform([l4_pred])[0]
l4_confidence = l4_probs.max()

if l4_pred_class == "action_actuation" and l4_confidence >= config.LEVEL4_CONFIDENT_THRESH:
    family_factor = config.LEVEL4_ACTION_ACTUATION_DOWNWEIGHT  # 0.3
else:
    family_factor = 1.0

w_adjusted = w * family_factor
```

Config constants: `config.LEVEL4_ACTION_ACTUATION_DOWNWEIGHT = 0.3` (blend weight multiplier
on action_actuation predictions), `config.LEVEL4_CONFIDENT_THRESH = 0.5` (minimum confidence to
apply the down-weight). Requires `level4_classifier.pkl` (pre-trained Level 4 classifier
artifact).

**Results.**

| Condition | v4 success | HX2 success | Delta | BH-adjusted p |
|---|---|---|---|---|
| `grip_state_falsification` | 14.0% | 18.2% | **+4.2pp** | **0.0034** |
| All other conditions | — | — | ~0 | ~1.0 |

**Confirmed win on target condition.** HX2 restores `grip_state_falsification` to statistical
parity with `sac_her` no-recovery (18.2% vs 19.8%, BH p=0.825) and produces a significant win
over plain v4 itself. This is the only confirmed positive result across all HX variants.
**Do-no-harm verification** (`recovery_do_no_harm_audit.csv`): HX2 is the only recovery variant
with zero confirmed harm across all 11 conditions.

### HX3 (v3) — Re-gate relocalization_expert on Level 4

**File:** `recovery/recovery_v4_hx3.py` (~110 lines)

**Motivation.** `object_pose_spoof` episodes were routed to the `relocalization_expert` (whose
weight is `spoofed_goal`), but the classifier almost never predicts `spoofed_goal` for those
episodes — it predicts `never_reached_object`. The relocalization expert was starved of
activation on exactly the episodes it was designed to fix.

**Implementation.** Sets the spoofed-goal class probability to the maximum of its classifier
value and the Level 4 `perception_state` probability:

```python
adjusted_class_probs["spoofed_goal"] = max(
    class_probs["spoofed_goal"],
    l4_probs["perception_state"]
)
```

Plain `max()`, no new config constant. The rest of `compute_recovery_action_hx()` is unchanged.

**Results.**

| Condition | v4 success | HX3 success | Delta vs v4 | BH p |
|---|---|---|---|---|
| `object_pose_spoof` | 28.2% | 26.0% | -2.2pp | 1.0 |
| `grip_state_falsification` | 14.0% | 19.1% | +5.1pp (vs v4) / +0.9pp (vs HX2) | 0.000034 / 1.0 |

**Genuine null on target.** The `object_pose_spoof` change is non-significant (BH p=1.0). HX3
preserves HX2's `grip_state_falsification` win (+5.1pp vs v4). Not adopted.

### HX4 (v4) — Full Expert Remap for Goal-Spoofing

**File:** `recovery/recovery_v4_hx4.py` (~192 lines)

**Motivation.** `goal_spoof_immediate` and `goal_spoof_midep` showed no recovery benefit in
HX3 — the routing fix didn't move the needle. HX4 tests a stronger hypothesis: that
`spoofed_goal` was assigned to the wrong expert entirely, and that the transport expert (which
reasons about object position relative to goal) is the right one for goal-spoofing.

**Implementation.** Full expert remap:

```python
adjusted_class_probs["transport"] = (
    class_probs["divergent_transport"] + class_probs["spoofed_goal"]
)
adjusted_class_probs["relocalization"] = l4_probs["perception_state"]
```

`scheduled_classes` for `transport_expert` changed from `["divergent_transport"]` to
`["divergent_transport", "spoofed_goal"]`. Both experts retain their original
stage-compatibility masks.

**Results.**

| Condition | v4 success | HX4 success | Delta vs v4 | BH p |
|---|---|---|---|---|
| `goal_spoof_immediate` | 10.2% | 10.4% | +0.2pp | 1.0 |
| `goal_spoof_midep` | 10.4% | 10.9% | +0.4pp | 1.0 |
| `grip_state_falsification` | 14.0% | 18.9% | +4.9pp | 0.000066 |

**Genuine null on target.** Goal-spoofing improvement is non-significant. HX4 preserves HX2's
`grip_state_falsification` win. Not adopted.

### HX5 (v5) — Global Fast-Attack Trigger EMA

**File:** `recovery/recovery_v4_hx5.py` (~141 lines)

**Motivation.** The investigation into why recovery shows no benefit on goal-spoofing shifted
from expert routing to trigger timing. Analysis: on `goal_spoof_immediate` (attacked from step
0), `w` doesn't cross the minimum activation threshold until step ~16, and is still only ~24%
strength by step 39. In a 150-step episode, losing the first 40–60+ steps to a diluted or
absent correction plausibly explains most of the gap.

**Implementation.** New class `FastAttackTriggerWeight(TriggerWeight)`. Overrides `update()`:

```python
if p_fail > self.ema_pfail:  # rising = new failure detected
    effective_alpha = self.alpha * config.RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER
else:
    effective_alpha = self.alpha  # original speed on falling side
```

`RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER = 4.0` (in `config.py`). Applies **globally** — the
fast EMA fires on all attack conditions when `p_fail` rises, not just goal-spoofing. Verified
trigger speedup: `w` crosses `EPSILON` by step 3–4 instead of step 16 on `goal_spoof_immediate`.
By step 39: `w` ≈ 0.65 (vs v4's 0.24).

**Results.**

| Condition | v4 success | HX5 success | Delta vs v4 | BH p |
|---|---|---|---|---|
| `goal_spoof_immediate` | 10.2% | 9.8% | -0.4pp | 1.0 |
| `goal_spoof_midep` | 10.4% | 12.2% | +1.8pp | 1.0 |
| `grip_state_falsification` | 14.0% | 16.7% | +2.7pp | 0.756 (raw p=0.058) |

**Genuine null on target, plus regression risk.** Goal-spoofing improvement is non-significant.
The `grip_state_falsification` win weakens from HX2's +4.2pp to HX5's +2.7pp (raw p=0.058, not
BH-significant but a soft signal of regression). The global speedup changes trigger behavior on
the one condition HX2 fixed. Not adopted.

### HX6 (v6) — Level-4-Gated Fast-Attack Trigger (FINAL)

**File:** `recovery/recovery_v4_hx6.py` (~153 lines)

**Motivation.** HX5's global speedup was too broad — it improved trigger timing on
goal-spoofing but degraded behavior on `grip_state_falsification` (an `action_actuation`
condition where the trigger should not change speed). The fix: apply HX5's fast EMA **only**
when Level 4 predicts an attack family that actually needs faster response.

**Implementation.** New class `GatedFastAttackTriggerWeight(TriggerWeight)`. Overrides
`update()` with two extra arguments (`l4_pred_class`, `l4_confidence`):

```python
if p_fail > self.ema_pfail:
    l4_attack_needs_speedup = (
        l4_pred_class in {"perception_state", "goal_manipulation"}
        and l4_confidence >= config.LEVEL4_CONFIDENT_THRESH
    )
    if l4_attack_needs_speedup:
        effective_alpha = self.alpha * config.RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER
    else:
        effective_alpha = self.alpha
else:
    effective_alpha = self.alpha
```

**Key structural change:** Level 4 classification runs **before** `trigger.update()` (in
HX2–HX5, it ran after). This ensures the fast-attack gate is set before the trigger computes
its new weight. The same `l4_probs` are reused for the action_actuation down-weight (from HX2).
Reuses `config.RECOVERY_V4_HX5_ATTACK_ALPHA_MULTIPLIER` and `config.LEVEL4_CONFIDENT_THRESH` —
no new config constants added.

**Results.**

| Condition | v4 success | HX6 success | Delta vs v4 | BH p |
|---|---|---|---|---|
| `grip_state_falsification` | 14.0% | 18.2% | **+4.2pp** | **0.0017** |
| `grip_state_falsification` vs HX2 | 18.2% | 18.2% | **0.000pp** | **1.0** |
| `goal_spoof_immediate` | 10.2% | 9.6% | -0.7pp | 1.0 |
| `goal_spoof_midep` | 10.4% | 11.3% | +0.9pp | 1.0 |

**Identical to HX2 on the one confirmed win** (delta=0.000, BH p=1.0). **No regression on any
condition.** **Do-no-harm vs sac_her is completely clean** (zero significant harm across all 11
conditions).

**Overall statistics** (from `final_hx_comparison_summary_table.csv`):

| Metric | HX | HX2 | HX3 | HX4 | HX5 | **HX6** | v4 | v2 | v3 |
|---|---|---|---|---|---|---|---|---|---|
| Overall success | 33.00% | 33.47% | 33.62% | 33.42% | 33.38% | **33.47%** | 33.39% | 34.0% | 33.62% |
| Clean performance | 100% | 100% | 100% | 100% | 100% | **100%** | 100% | 100% | 100% |
| Safety violation rate | 0.13% | 0.09% | 0.11% | 0.09% | 0.11% | **0.06%** | 0.06% | 2.6% | 4.0% |

**Verdict: adopted as the final controller.** Strict superset of HX2 — preserves its one
confirmed win (bit-for-bit identical), zero regression risk, adds the gated fast-trigger for
future use if a Level 4 signal becomes strong enough on goal-spoofing conditions.

### Summary: All Confirmed Significant Results

| Variant | Condition | Success Rate | Improvement vs v4 | BH-adjusted p |
|---|---|---|---|---|
| **HX2** | `grip_state_falsification` | 18.2% | +4.2pp | 0.0034 |
| **HX3** | `grip_state_falsification` | 19.1% | +5.1pp | 0.000034 |
| **HX4** | `grip_state_falsification` | 18.9% | +4.9pp | 0.000066 |
| **HX6** | `grip_state_falsification` | 18.2% | +4.2pp | 0.0017 |

**HX3 and HX4's grip_state wins are not independent of HX2** — they inherit HX2's down-weight
mechanism and add orthogonal changes. The only independently confirmed improvement across all
variants is HX2's (now adopted in HX6).

### v2/v3 vs. v4-HX Family Tradeoff

From `final_hx_comparison_timing.csv` and `final_hx_comparison_success_safety.csv`:

| | v2/v3 baseline | v4/HX6 |
|---|---|---|
| **Trigger mechanism** | Hard-threshold (3 rule-based signals) | Continuous blend (classifier-driven) |
| **Override style** | Full, unblended | Gradual ramp |
| **Response time** | ~9 steps to activation | ~14–16 steps to activation |
| **Time to half-strength** | 0 steps (instant full override) | ~66–77 steps |
| **C4 safety violation rate** | 2.6% / 4.0% | 0.1% / 0.2% |
| **Goal-spoofing improvement** | +2.9pp / +4.9pp | None |

**The tradeoff:** v2/v3 trade safety for speed — they detect and override faster, but their
full unblended override causes more safety violations. v4/HX6 trade speed for safety — they
ramp in gradually, which is safer but means they lose the first 40–60+ steps of the episode to
a diluted or absent correction. This tradeoff is architectural, not tunable within either
framework.

### Bottom Line

1. **HX2 fixed a real harm** in plain v4 — the only independently confirmed positive result
   across all variants.
2. **HX3, HX4, HX5, HX6** systematically investigated remaining gaps (object pose spoofing,
   goal spoofing, trigger speed), all returned genuine nulls on their targets.
3. **HX6 is adopted** as a safe superset of HX2 — preserves its one confirmed win, zero
   regression risk, adds the gated fast-trigger for future use.
4. **The goal-spoof gap is architectural** — v3's hard-override achieves +2.9pp/+4.9pp there
   because it sacrifices safety; v4's continuous blend achieves 0% there because it prioritizes
   safety. No variant of v4 closes this gap without reintroducing safety violations.

*(n=450, seeds 0–14, all 11 attack conditions, clean_2M PickAndPlace checkpoint;
Benjamini-Hochberg FDR-corrected across variant comparisons.)*

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
├── paper/                       # Paper draft (LaTeX, figures) — local only, gitignored
├── policies/                    # Rule-based and SAC+HER policy wrappers
├── recovery/                    # v2 (step-at-a-time), v3 (sustained window),
│                                #   v4 (CCAR), v4_hx..v4_hx6 (TAIRO-HX variants —
│                                #   v4_hx6 is the adopted variant)
├── scripts/                     # Sweep entrypoint, figure builders, training scripts,
│                                #   diagnostic/calibration scripts —
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
└── TAIRO-HX.md                  # The five-level hierarchical failure-diagnosis stack
```

> A number of other project-tracking `.md` docs (design history, phase-by-phase findings,
> session handoffs, paper-writing notes) exist locally for development continuity but are
> gitignored and intentionally not part of the public repo.

> **Note:** an earlier version of this structure listed a top-level `notebooks/` directory that
> no longer appears in the repo file listing — worth confirming whether it was removed
> intentionally or should be restored/gitignored explicitly.

The `results/` tree is fully gitignored except for episode-result CSVs (not the much larger
per-step logs) across `results/data_recovery_v4*/` — this includes the Phase 6 evaluation,
the dense sweep, every TAIRO-HX variant's evaluation, the v2/v3 backfill, and the seeds-5-14
power-check directories — plus `results/archive/README.md` and small metrics/summary CSVs
under `results/classifier_level1/`, `results/classifier_level4/`.
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

## Experimental History

The project moved through several phases: baseline SAC+HER evaluation across all 11 attack
conditions, a failure-mode classifier workstream (a six-label taxonomy plus an online/causal
classifier), the Recovery v2/v3 hard-threshold controllers, Recovery v4's classifier-conditioned
adaptive blend (CCAR), and finally the TAIRO-HX hierarchy (task stage → anomaly → failure type
→ attack family → recovery decision) feeding into the adopted `v4-HX6` controller. A scoped,
separate attack-aware policy track (a ground-truth attack-category flag injected into SAC+HER
training) was explored but not carried through to completion. Detailed phase-by-phase design
and experiment notes are kept as local development records outside the public repo.
