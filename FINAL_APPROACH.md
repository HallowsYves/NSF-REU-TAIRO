# TAIRO's Final Recovery Approach — What It Is and How It Improves on the Baselines

*Written 2026-07-23 for the mentor's final-push checklist ("a clear explanation
of the final approach and its improvement over the baselines"). Standalone
document — doesn't assume you've read `RECOVERY_V4.md` or `findings.md`
first, though both have more raw detail if you want it. All numbers are
n=450 (seeds 0-14, all 11 attack conditions), Benjamini-Hochberg
FDR-corrected, `clean_2M` PickAndPlace checkpoint.*

---

## 1. The problem this solves

SAC+HER, trained cleanly, is vulnerable to a range of cyber-physical
attacks on a robot manipulation task (FetchPickAndPlace — grasp an object,
move it to a target). TAIRO's recovery system tries to detect that an
attack is degrading the robot's behavior and intervene online, without
knowing in advance which of 11 attack conditions is active.

Three generations of recovery controller were built and are compared here:

| Generation | Mechanism | Status |
|---|---|---|
| **v2 / v3** | Hard-threshold trigger: 3 rule-based signals (action-norm saturation, stalled progress, worsening distance) fire a **full, unblended override** for a sustained window once tripped. | Earlier baselines |
| **v4 (CCAR)** | Classifier-Conditioned Adaptive Recovery: a trained failure-mode classifier's output drives a **continuous blend weight** `w` between the original policy action and one of five expert recovery controllers. | Superseded by v4-HX6 |
| **v4-HX6 (adopted)** | v4's continuous blend, plus two hierarchical-diagnosis refinements (below). | **Final** |

---

## 2. What's new in the final approach

v4-HX6 = plain v4's CCAR mixture, **plus two additions**, each built to fix
a specific, measured problem — not speculative improvements.

### Addition 1: Level 4 attack-family down-weight (inherited from v4-HX2)

**The problem it fixes.** A systematic "do-no-harm" audit — comparing every
recovery method against doing nothing at all, across all 11 conditions —
found that plain v4 does **statistically significant harm** on
`grip_state_falsification` relative to no recovery: 19.8% → 14.0% success
(BH-adjusted p=0.0002). This was a genuine defect in the paper's own
then-current headline method, not a hypothetical concern.

**Why it happens.** `grip_state_falsification` and the other
`action_actuation`-family attacks (action clipping/delay/reversal) corrupt
the *command channel itself*. A recovery expert that reasons about
*state* (object position, gripper aperture, distance to goal) has no real
leverage there — blending its output in just adds noise on top of an
already-corrupted actuation path.

**The fix.** TAIRO-HX's Level 4 (attack-family) classifier predicts which
of 4 attack families is active from causal features alone (without seeing
the ground-truth condition). When it confidently predicts
`action_actuation`, the final blend weight is down-weighted
(`w × 0.3`) — the recovery expert still runs, but its influence on the
executed action is deliberately reduced for exactly the attack family
where it was shown to hurt.

**Confirmed result.** This addition alone (v4-HX2) restored
`grip_state_falsification` to **statistical parity with doing nothing**
(18.2% vs. 19.8%, not significantly different, p=0.92) and produced a
significant win over plain v4 itself (**+4.2pp, BH p=0.0017**).

### Addition 2: Level-4-gated fast-attack trigger (new in v4-HX6)

**The problem it fixes.** A separate investigation into why recovery shows
no measurable benefit on goal-spoofing conditions (`goal_spoof_immediate`/
`goal_spoof_midep`) — despite the older v3 baseline getting a real
+2.9pp/+4.9pp lift there — found the root cause in v4's own trigger
timing. The classifier-driven blend weight `w` uses a slow exponential
moving average: on `goal_spoof_immediate` (attacked from step 0), `w`
doesn't cross even its minimum activation threshold until step ~16, and is
still only ~24% strength by step 39. In a 150-step episode, losing the
first 40-60+ steps to a diluted or absent correction plausibly explains
most of the gap.

**Three earlier attempts, all genuine nulls:**
- **v4-HX3**: re-routed which expert consumes the `object_pose_spoof`
  signal. No confirmed change.
- **v4-HX4**: fully remapped goal-spoofing's expert assignment. No
  confirmed change.
- **v4-HX5**: sped up the trigger's reaction time globally (4× faster EMA
  on new failures). No confirmed change on goal-spoofing, **and** a soft,
  unconfirmed regression risk on the `grip_state_falsification` win from
  Addition 1 — because speeding up the trigger *everywhere* also changed
  its behavior on that condition, which it was never meant to touch.

**The v4-HX6 fix.** Apply hx5's same fast-trigger idea, but **gate it on
Level 4's own attack-family prediction** — the 4× faster EMA only engages
when Level 4 confidently predicts `perception_state` or
`goal_manipulation` (the families the investigation actually targeted).
On `action_actuation` conditions like `grip_state_falsification`, the
trigger keeps its original, unmodified speed — so the fast trigger
*cannot* interfere with Addition 1's down-weight, by construction rather
than by retuning.

**Confirmed result.** At full power: still a genuine null on the
goal-spoofing target itself (no confirmed movement — the fourth
independent hypothesis to come back negative on that specific problem).
But `grip_state_falsification` is **exactly, bit-for-bit identical to
v4-HX2** (delta = 0.000, p = 1.0) — the gate works precisely as designed —
and the do-no-harm audit against `sac_her` is **completely clean**, with
zero significant harm anywhere, unlike v4-HX5's regression risk.

**Why it's adopted anyway.** v4-HX6 is a strict improvement over v4-HX2 in
this data: identical or better on every measured metric, with a more
robust trigger design and zero new regression risk. This is a different
kind of adoption argument than every prior variant in this project (all of
which required a *new confirmed win* to be adopted) — flagged explicitly
rather than applied silently, since "no regression + more robust design"
is a real but different bar than "confirmed benefit."

---

## 3. How much does the final approach actually improve on the baselines?

This is the part worth being completely honest about: **it's not a clean
win across the board.** The full comparison below is no-recovery vs.
earlier baselines (v2/v3) vs. gradual-response CCAR (v4) vs. final
selective recovery (v4-HX6), all at matched n=450 power.

### 3.1 Where v4-HX6 is a genuine, confirmed improvement

- **Fixes a real regression in its own architecture.** Plain v4 harms
  `grip_state_falsification` (−5.8pp vs. no recovery, p=0.0002). v4-HX6
  corrects this to a **+4.2pp win over plain v4** (p=0.0017) and restores
  parity with doing nothing.
- **Beats the earlier v3 baseline on two conditions**:
  `action_clipping` (+3.1pp, p=0.03) and `action_delay` (+4.7pp, p=0.03).
- **Zero confirmed harm anywhere** across the full 11-condition grid vs.
  `sac_her` — the only method in the whole comparison set with a
  completely clean do-no-harm record at this sample size.
- **Far safer than the earlier baselines.** C4 safety-violation rate is
  0.06% overall (essentially zero), vs. 2.6% (v2) and 4.0% (v3) — both
  *higher* than doing nothing at all (0.26%). On `grip_state_falsification`
  specifically, where the earlier baselines fire most often, the gap is
  stark: v2 hits a 9.1% violation rate, **v3 hits 19.6%** — nearly 1 in 5
  attacked episodes — while v4-HX6 stays at 0.2%.

### 3.2 Where the earlier baselines still win — stated plainly

- **On `grip_state_falsification` itself — the one condition with any
  confirmed recovery effect at all — the older v2/v3 baselines
  significantly outperform v4-HX6**: v2 beats it by +8.7pp
  (p=0.00001), v3 by +7.8pp (p=0.0001).
- **Why**: recovery latency. v2/v3 detect a failure in ~9 steps and apply
  a full, unblended correction on that same step (zero response delay by
  design). v4-HX6 detects more slowly (~14-20 steps) *and* ramps up
  gradually — another ~65-80 steps to reach half-strength blend authority,
  ~87-92 total steps before the correction is at full strength, in a
  150-step episode. The gated fast-trigger addition helps close *some* of
  this gap on average (v4-HX6's mean total latency, 87 steps, edges out
  plain v4's 92) but not on `grip_state_falsification` itself, since the
  gate deliberately excludes that condition.

### 3.3 The honest one-sentence summary

**v4-HX6 is the safest recovery architecture tested, and the only one with
a completely clean do-no-harm record — but on the single condition where
recovery demonstrably helps at all, the older, faster, less-safe hard-override
baselines still win outright.** Speed and safety trade off across recovery
*architectures*, not just within one — that's a real finding about how
recovery systems should be designed and scored, not a caveat to bury.

---

## 4. All 8 requested metrics, one table

| Metric | No recovery | v2 | v3 | v4 (gradual) | **v4-HX6 (final)** |
|---|---|---|---|---|---|
| Task-success rate (overall) | 33.5% | 34.0% | 33.6% | 33.4% | 33.5% |
| Clean-task performance | 99.8% | 99.8% | 99.1% | 99.8% | **100.0%** |
| Detection delay (steps) | — | 9.3 | 9.5 | 15.9 | 20.0 |
| Recovery-response delay (steps) | — | 0.0 | 0.0 | 76.1 | 66.5 |
| Recovery time (steps active) | — | 5.4 | 11.1 | 100.3 | 95.5 |
| Safety-violation rate | 0.26% | 2.6% | 4.0% | 0.06% | **0.06%** |
| Number of interventions | — | 1.16 | 1.07 | 0.86 | 1.17 |
| Completion-time overhead (steps) | — | +0.06 | +2.76 | −1.65 | −0.40 |

*(Full breakdown by condition: `results/final_hx_comparison_summary_table.md`;
raw comparisons: `results/final_hx_comparison_{success_safety,delays,timing,final_head_to_head}.csv`.)*

Task-success rate and completion-time overhead are essentially flat across
all methods when averaged over all 11 conditions — that's expected and not
a red flag: most conditions are either untouched by any recovery method
(structural failures like `sensor_dropout`/`action_reversal`, ~3% success
everywhere) or already near-ceiling (clean, `action_clipping`). The real
differentiation is concentrated in the handful of conditions shown in
§3, which is exactly why the per-condition breakdown (Fig. 1) matters more
than the pooled average.

---

## 5. What this means going forward

- **v4-HX6 is the adopted final controller** — wired into
  `evaluation/episode_runner.py`, `scripts/run_multiseed_sweep.py`, and the
  live Streamlit demo (`app/live_attack_demo.py`).
- **Four independent, well-powered negative results on goal-spoofing**
  (v4-HX3/4/5/6) is itself a real finding: the gap between v3's
  hard-override and the v4-family's continuous blend most likely reflects
  an *architectural* difference (instant full correction vs. gradual
  ramp), not a fixable routing or timing defect inside the current CCAR
  design. Worth stating in the paper as a documented limitation, not
  hidden.
- **Open question, not resolved here**: whether a future design should
  make the *architecture choice itself* — hard override vs. continuous
  blend — conditional on the diagnosed attack family, rather than treating
  "continuous blend" as fixed and only tuning within it. That's a bigger
  change than anything tried in this investigation and is flagged for
  discussion, not attempted.
