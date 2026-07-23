# TAIRO — Poster Draft

*Draft content for the IEEE BigData 2026 REU Symposium poster. Written as
panel-by-panel copy ready to move into Figma — prose is kept tight on
purpose; expand/trim per the actual poster's real estate. Numbers current
as of 2026-07-23 (n=450, seeds 0-14, all 11 conditions, `clean_2M`
PickAndPlace checkpoint). `sac_her_recovery_v4_hx6` is the adopted final
controller — see `FINAL_APPROACH.md` for the complete standalone
explanation of the approach and its improvement over baselines.*

---

## Title panel

**TAIRO: Trustworthiness and Adversarial Impact on Robotic Operations**
*A hierarchical failure-diagnosis and adaptive-recovery framework for
robot manipulation under cyber-physical attack*

Yves Velasquez Vega (CSU Fullerton), Jachin Choi (Case Western Reserve
University), Sunny Sood, Abhinav Kochar (University of Missouri-Kansas
City) — advised by Dr. Duy Ho (CSUF), Dr. Yugyung Lee (UMKC)
NSF REU Site "AI-empowered Cybersecurity," Grant CNS-2349236

---

## Motivation panel

- Robot manipulation policies (SAC+HER) trained under clean conditions are
  vulnerable to a wide range of cyber-physical attacks — sensor spoofing,
  action-channel corruption, goal manipulation — and existing benchmarks
  mostly measure *how much* success degrades, not whether a system can
  *detect and recover* online.
- TAIRO evaluates SAC+HER under an 11-condition attack taxonomy on
  FetchPickAndPlace-v4 (object grasp-and-place), scored on a five-component
  trustworthiness lens (C1–C5: reliability, robustness, cyber resilience,
  safety, recovery) rather than raw success rate alone.
- This poster focuses on the recovery side: **can an online failure-
  diagnosis hierarchy make recovery genuinely better, not just detect that
  something went wrong?**

---

## Approach panel

**Attack taxonomy.** 11 conditions spanning sensor, action, and goal
channels, plus 3 manipulation-specific attacks (object-pose spoofing,
gripper-state falsification, contact dropout).

**Recovery architecture — three generations, compared head-to-head:**
1. **v2/v3 (earlier baselines).** Rule-based, hard-threshold trigger: 3
   detection signals (action-norm saturation, insufficient progress,
   distance-trend) fire a full, unblended rule-based override for a
   sustained window.
2. **v4 — gradual-response CCAR** (Classifier-Conditioned Adaptive
   Recovery). Replaces the hard trigger with a continuous blend weight `w`
   driven by an online failure-mode classifier's EMA-smoothed `p_fail`,
   mixing five feature-based expert controllers into the policy action.
3. **v4-HX6 — final, selective recovery (adopted).** Layers TAIRO-HX's
   hierarchical diagnosis (Task Stage → Anomaly → Failure Type → Attack
   Family → Recovery Decision) back into v4's mixture: Level 1 stage-gates
   which expert can fire, Level 4 down-weights the blend when it
   confidently detects an `action_actuation`-family attack (action
   clipping/delay/reversal, gripper falsification) — channels where
   *state-based* recovery has no real leverage and was shown to actively
   hurt — and additionally speeds up the failure-detection trigger, but
   ONLY when Level 4 confidently detects a `perception_state`/
   `goal_manipulation`-family attack, so the speed-up can never interfere
   with the action_actuation down-weight above.

**Why each piece exists.** A systematic do-no-harm audit found plain v4 —
the architecture's own starting point — does *significant harm* relative
to no recovery at all on `grip_state_falsification` (19.8% → 14.0% success,
BH-adjusted p=0.0003). The Level 4 down-weight was built specifically to
fix this. A separate investigation into weak recovery on goal-spoofing
conditions traced the gap to the trigger's own reaction speed (~16-60+
steps to ramp up); the gated fast-attack trigger targets that lag directly
without reopening the down-weight's own tuning.

---

## Key Results panel

**Figure 1 — success rate by condition, all 4 arms**
`results/figures/final_hx_comparison/fig1_success_by_condition.png`

**Figure 2 — recovery latency**
`results/figures/final_hx_comparison/fig2_recovery_latency.png`

**Headline numbers (all n=450, BH-FDR corrected):**

| Comparison | Condition | Δ | p (BH) |
|---|---|---|---|
| v4-HX6 vs. plain v4 | grip_state_falsification | **+4.2pp** (14.0%→18.2%) | 0.0017 |
| v4-HX6 vs. sac_her (no recovery) | grip_state_falsification | −1.6pp (parity, not harm) | 0.92 |
| v2 vs. v4-HX6 | grip_state_falsification | **+8.7pp** (v2 wins) | 0.00001 |
| v3 vs. v4-HX6 | grip_state_falsification | **+7.8pp** (v3 wins) | 0.0001 |
| v4-HX6 vs. v3 | action_clipping / action_delay | **+3.1pp / +4.7pp** (v4-HX6 wins) | 0.03 / 0.03 |

**The honest headline is a tradeoff, not a clean ranking:**
- v4-HX6 **fixes** plain v4's regression and restores parity with doing
  nothing — genuine, statistically confirmed (inherited unchanged from its
  predecessor v4-HX2's mixture).
- The **older v2/v3 baselines still outright beat v4-HX6** on
  `grip_state_falsification`, the one condition with any confirmed effect
  at all — because their hard-override responds in **~9 steps**, while
  v4-HX6's continuous blend takes **~87-92 steps** to reach full strength
  (Fig. 2).
- v4-HX6's advantage is **safety**: C4 safety-violation rate 0.06%, vs.
  2.6% (v2) / 4.0% (v3) — both *higher* than doing nothing at all (0.3%).
  Fast hard-override recovery costs safety; v4-HX6's gradual blend doesn't.

**All 8 tracked metrics, one row per arm** (task-success rate, clean-task
performance, detection delay, recovery-response delay, recovery time,
safety violations, number of interventions, completion-time overhead):
`results/final_hx_comparison_summary_table.md`.

---

## Conclusion panel

- **`sac_her_recovery_v4-HX6` is the adopted final recovery controller** —
  the only variant that both fixes a real regression in its own
  architecture's baseline *and* keeps that architecture's safety
  advantage over the alternative (hard-override) design, with zero
  confirmed regression anywhere across the full 11-condition grid.
- **Speed and safety trade off across recovery architectures**, not just
  across individual fixes within one architecture — a finding that
  matters for how "recovery quality" should be scored, not just whether a
  given variant beats its own predecessor.
- Four independent, well-powered follow-up hypotheses for closing the
  remaining goal-spoofing gap (wrong expert routing, weak classifier
  signal, slow trigger EMA, gated fast trigger — `v4_hx3`/`v4_hx4`/`v4_hx5`/
  `v4_hx6`) were tested and are documented negative results on that
  specific target, not oversights — see `RECOVERY_V4.md` for the full,
  transparent record. `v4_hx6` was still adopted despite its own null,
  because it is a strict safety/robustness improvement over `v4_hx2` with
  zero measured downside — an explicit, deliberate departure from this
  project's usual "adopt only on a new confirmed win" bar, flagged as such.
