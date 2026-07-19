# reported_grasp / contact — Definitional Options

**DECISION (2026-07-18): Option A adopted.** Keep the causal proxy as-is — no forward lift
check, documented as an intentionally conservative online signal. Rationale: the disagreement
is small (4.61%, 304/6,600 episodes) and strictly one-directional (the proxy never falsely
claims a grasp the hindsight labeler doesn't also confirm) — the safe failure mode for a signal
feeding failure classification and recovery decisions. Option B (lift-trend check) only helps
at dense classifier cadence, which hasn't been adopted yet (open since Recovery v4 Phase C).
Option C (splitting into two signals) adds schema complexity for an edge case that's partly
explained by legitimate clean-episode dragging behavior anyway. Documented in-code at
`evaluation/causal_features.py`'s `grasp_kinematic_ever_sofar` computation and in
`findings.md`'s "Grasp detection" section / `CLAUDE.md` §12. The analysis below is kept intact
as the record of why.

**Status:** decision document, resolved. Not a governing doc. No code in `causal_features.py`,
`failure_mode_labeling.py`, or any taxonomy definition was changed by the analysis itself — only
the comment/doc pointers noted above were added once the decision was made.
Produced 2026-07-18 on branch `investigate/ws2-metric-and-grasp-def` as TAIRO-HX Phase 0.5
Phase B.

## The gap

`evaluation/causal_features.py`'s online/causal proxy (`grasp_kinematic_ever_sofar`) confirms
a grasp when two kinematic conditions — in-contact (`distance_to_object < GRASP_DIST_THRESHOLD`)
and co-moving (`‖object_velp − grip_velp‖ < GRASP_TRACKING_THRESHOLD`) — hold for `GRASP_WINDOW`
(5) consecutive steps. It deliberately has **no forward-looking lift check**, because a lift
check needs steps after the window closes, which violates the no-hindsight constraint for an
online classifier.

`evaluation/failure_mode_labeling.py`'s hindsight labeler (`_detect_grasp`) requires the same
two kinematic conditions **plus** a subsequent Z-lift (`object_pos_z` rises ≥
`GRASP_LIFT_THRESHOLD` within `GRASP_LIFT_WINDOW` steps after the kinematic window closes) —
this is what actually distinguishes a real grasp from dragging the object across the table.

Every episode where hindsight confirms a grasp necessarily satisfies the causal proxy's
kinematic condition too (lift is an additional requirement on top of kinematic, not an
alternative path) — confirmed empirically below: zero episodes go the other way. So the only
possible disagreement direction is: **causal proxy says "grasped," hindsight says "reached
but never actually grasped — dragging."**

## Quantified disagreement (results/classifier_seedfix/, 6,600 episodes, all 4 models)

**Raw causal proxy (currently implemented, no lift check):**

- **304 / 6,600 episodes disagree (4.61%)**, 100% in the single expected direction
  (causal=confirmed, hindsight=dragging). Zero episodes disagree the other way.
- All 304 disagreement episodes carry the hindsight label `reached_but_failed_grasp` — this
  gap is entirely internal to that one label; it never bleeds into `success`,
  `grasped_but_dropped`, or any other class.
- **Concentrated, not uniform** — and *not* mainly in sensor-related attacks:

  | Condition | Disagreement | Worst single (condition, model) |
  |---|---|---|
  | `action_clipping` | 10.50% | `randomized_2M`: 21.33% |
  | `clean` | 6.83% | `clean_2M`: 12.67% |
  | `object_pose_spoof` | 6.33% | `clean_2M`: 11.33% |
  | `action_delay` | 6.17% | `randomized_2M`: 9.33% |
  | `grip_state_falsification` | 6.17% | `clean_2M`: 18.00% |
  | `goal_spoof_midep` | 5.00% | — |
  | `goal_spoof_immediate` | 4.50% | — |
  | `sensor_bias` | 2.83% | — |
  | `sensor_dropout` | 1.17% | — |
  | `action_reversal` | 0.67% | — |
  | `contact_dropout` | 0.50% | — |

  The highest disagreement is in **action-space conditions** (`action_clipping`,
  `grip_state_falsification`) and, notably, **`clean`** itself (6.83%, 12.67% for `clean_2M`
  specifically) — meaning roughly 1 in 8 clean-condition `clean_2M` episodes that "reach and
  hold contact" are actually dragging, not lifting, even with no attack active. This is a
  real property of the policy's grasping behavior, not an attack artifact. The lowest
  disagreement is in the conditions where the policy essentially fails outright before it
  ever gets near the object (`contact_dropout`, `action_reversal`, `sensor_dropout`) — there's
  little grasp ambiguity to disagree about when contact rarely happens at all.

## Exploratory: can an online-computable lift-trend check close the gap?

Tested a purely backward-looking approximation — once the kinematic streak first fires, check
whether `obj_z_now − min(obj_z over the trailing CAUSAL_WINDOW_SHORT=20 steps)` ever exceeds
`GRASP_LIFT_THRESHOLD` afterward. This uses only past/current data at each query point, so it's
legitimately causal — **not implemented anywhere, offline check only.**

**Result depends heavily on query cadence, which matters for the decision:**

| Cadence tested | Disagreement | FP (over-claims grasp) | FN (misses real grasp) |
|---|---|---|---|
| Sparse, 10-step stride (matches the **production** `online_failure_classifier.pkl`'s training cadence) | **4.88%** (322/6,600) — *no better than doing nothing* | 0.32% | 4.56% |
| Dense, every step (matches `classifier_seedfix_dense/`, not yet the production cadence) | **1.20%** (79/6,600) — **~3.8× reduction** | 0.44% | 0.76% |

At the sparse cadence the production classifier actually uses today, checking every 10th step
misses more real lifts than it correctly rules out drags — it's a net wash, arguably worse
since it shifts error from a one-directional, well-understood bias (Option A) to a two-sided,
less-predictable one. At every-step (dense) cadence, it's a genuine and fairly large
improvement. **The viability of Option B is gated on whether the deployed path queries dense
or sparse — this is an existing open cadence question (see the Phase C dense-classifier work
in `RECOVERY_V4.md`), not something this investigation can resolve unilaterally.**

Remaining dense-cadence disagreement is still concentrated in `clean` (3.50%) and
`object_pose_spoof` (2.33%) — plausibly a window-alignment artifact (the causal check looks
for a rise "at any point after onset," unbounded, vs. the hindsight labeler's strict
`GRASP_LIFT_WINDOW`-bounded post-onset window), not an attack-specific effect.

## Options

**Option A — Keep the causal proxy as-is; document it explicitly as a deliberately looser
online signal, not a lift-verified grasp.**
Zero engineering cost, zero risk of regression. Affects 4.61% of episodes (304/6,600),
100% in the conservative direction (never denies a real grasp, occasionally credits a drag).
Concentrated in `action_clipping` (10.5%) and `grip_state_falsification` (6.2%), but also
present in `clean` (6.8%) — so the documentation should be explicit that this is partly a
property of `clean_2M`'s own grasping behavior, not purely an attack-detection gap. Best fit
if downstream consumers (e.g. Recovery v4's `grasp_stabilize_expert`/`regrasp_expert`
weighting) can tolerate an occasional false "grasp confirmed" without much cost.

**Option B — Add the online-approximate lift-trend check, accepting the cadence tradeoff as a
known dependency.**
Only worth doing **if paired with dense (every-step) querying** — at that cadence it cuts
disagreement from 4.61% to 1.20% (roughly 3.8×) with a reasonably balanced error profile. At
the current production sparse cadence it doesn't help (4.88%, arguably worse-shaped than
Option A). This option is really "Option B contingent on the sparse-vs-dense classifier-cadence
decision" — it should be evaluated together with that decision, not independently.

**Option C — Keep `grasp_kinematic_ever_sofar` (contact) and a lift-trend signal as two
separate exposed fields, rather than trying to collapse them into one "grasp confirmed"
boolean.**
The data suggests these measure genuinely different things: kinematic streak ≈ "is the
gripper in sustained contact and moving with the object" (a *contact* signal, useful on its
own for e.g. detecting `grip_state_falsification`), while the lift trend ≈ "is that contact
turning into an actual pick" (a *grasp* signal). Rather than picking one threshold that has to
serve both purposes, expose `contact_streak_now`/`grasp_kinematic_ever_sofar` as the contact
signal and (if dense querying is adopted) the lift-trend check as a separate, explicitly
weaker-confidence grasp signal — letting downstream consumers (recovery experts, TAIRO-HX's
proposed Task Stage / Failure Type layers) choose which one they actually need instead of
inheriting one proxy's blended semantics. This avoids forcing the cadence decision in Option B
to be resolved before any of this can be documented or used.

## Recommendation framing (not a decision — for the mentor conversation)

If a same-day answer is needed: **Option A** is the safe default — it's already what's
running, the gap is small, one-directional, and now precisely characterized. **Option B/C**
are worth revisiting together with the sparse-vs-dense production-classifier cadence decision
that Recovery v4's Phase C work already raised — bundling both cadence-sensitive decisions into
one conversation avoids re-deriving this tradeoff twice.
