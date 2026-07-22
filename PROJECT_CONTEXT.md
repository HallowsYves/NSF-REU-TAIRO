# TAIRO (NSF-REU-TAIRO) — Context

## Overview
- Goal: TAIRO evaluates SAC+HER robot manipulation policies under 11 adversarial attack conditions, producing a five-component trustworthiness score (C1–C5). Currently scoping TAIRO-HX, a proposed hierarchical failure-diagnosis redesign (Task Stage → Anomaly Detection → Failure Type → Attack Family → Recovery Decision).
- Constraints or deadlines: IEEE BigData 2026 REU Symposium submission target.

## Current Status
**Mentor feedback received 2026-07-22: paper writing is paused — focus fully on producing and
validating experimental results.** Poster and video demonstration are the near-term
deliverables, not the paper draft. Explicit ask: present the stage-aware (`v4_hx`),
attack-aware (`v4_hx2`), and routing-fix (`v4_hx3`) recovery results with tables, charts,
statistical comparisons, and brief interpretations, and continue pursuing additional recovery
improvements beyond what's been found so far ("we need strong, concrete results before we can
finalize the paper"). This supersedes the prior session's "shift to paper writing" plan below —
that framing is left in place as accurate history of what was decided *before* this feedback
arrived, not as the current plan (see Next Steps for the current one).

TAIRO-HX Levels 1-5 are all built. Level 1 (task stage): labels + classifier complete. Level 2 (anomaly detection): complete, `clean_2M`-scoped. Level 3 (behavioral failure): served by the existing Phase 8/9 classifier; multi-label extension re-raised 2026-07-20 and reconfirmed deferred. Level 4 (attack family): ground-truth labels + standalone classifier complete. **Level 5 (Recovery Decision): built and evaluated, closed with a documented, unfixable limitation** — its `stop_safely` proxy false-fires on 71.9% of recoverable episodes; a calibration sweep confirmed this is a genuine information-theoretic ceiling at 14-checkpoint granularity, not a tuning problem (`findings.md` Phase 10). Level 5 stays offline-only, not wired into any runtime controller.

**Mentor redirect (2026-07-20): the hierarchy should genuinely improve recovery, not stay a parallel offline track — this has now been acted on with real, power-checked results (2026-07-21).** Two new recovery variants were built on top of `recovery_v4.py`'s existing Tier 1 CCAR (which itself is untouched):
- **`sac_her_recovery_v4_hx`** (Level 1 stage-gating alone): evaluated at full power (n=450, all 11 conditions). No confirmed benefit anywhere. **Should not be adopted.**
- **`sac_her_recovery_v4_hx2`** (Level 1 + Level 4 attack-family down-weight): **the keeper.** Real, BH-corrected-significant win on `grip_state_falsification` (+4.2pp, 14.0%→18.2%, BH-adjusted p=0.0034), robust across the full 11-condition grid, not just the initially-flagged 4-condition subset.

**New finding from a systematic do-no-harm audit** (every recovery method vs. plain `sac_her`, 55 comparisons, BH-FDR corrected — `scripts/audit_recovery_do_no_harm.py`, `results/recovery_do_no_harm_audit.csv`): **plain `sac_her_recovery_v4` — the paper's current headline B4 method — does statistically significant harm on `grip_state_falsification` relative to no recovery at all** (19.8%→14.0%, BH-adjusted p=0.00017). `v4_hx` inherits this unchanged; `v4_hx2` is what corrects it back to statistical parity with doing nothing (18.2% vs 19.8%, not significantly different). No other method/condition pair in the grid shows confirmed harm or benefit. This reframes v4_hx2 as fixing a real problem in the currently-paper-facing method, not just improving on an already-fine baseline — and raises a question (not yet decided) about whether the paper's existing Tier 1 CCAR section needs a caveat on this condition regardless of whether v4_hx2 itself is adopted. Full detail: `RECOVERY_V4.md` §5.3, §5.6, §5.7; `findings.md` Phase 11 + its 2026-07-21 update.

Prior Item 1 result for reference (unchanged): flat RF 0.9424/0.8159 (canonical) < flat XGBoost 0.9485/0.8240 < hierarchical RF 0.9568/0.8734 < hierarchical XGBoost 0.9614/0.8817 (accuracy/macro-F1, episode-level, seed-4 test).

**One bounded follow-up fix, tested and closed (2026-07-22): `sac_her_recovery_v4_hx3`** re-gated `relocalization_expert` (built for `object_pose_spoof` but near-inert since it's keyed to the wrong classifier signal) on Level 4's `perception_state` probability instead. User explicitly scoped this as one bounded fix before shifting to paper writing, evaluated immediately at full power (n=450, all 11 conditions, avoiding the earlier underpowered-null mistake). **Result: a genuine, well-powered null** — no confirmed improvement on `object_pose_spoof` vs. either `v4` or the adopted `v4_hx2` (point estimate even moved slightly the wrong way, not significantly), but also no new harm anywhere, and it preserves `v4_hx2`'s `grip_state_falsification` win unchanged. **`v4_hx2` remains the sole adopted variant — `hx3` should not replace it.** Full detail: `RECOVERY_V4.md` §5.9. This closes the recovery-integration work for now per the user's own scope choice; next step is paper writing.

**New, 2026-07-21: an interactive Streamlit live-attack demo was built** (`app/live_attack_demo.py`, `app/sim_worker.py`) — the SAC+HER `clean_2M` PickAndPlace policy runs live against user-selected attack conditions/magnitudes, with a real-time MuJoCo render on one side and playback controls (Start/Pause/Step/Reset) on the other. Committed as `71e03bc streamlit implementation v1`. This is a separate, parallel workstream from the TAIRO-HX hierarchical classifier work above — not part of the recovery-integration plan.

**2026-07-21: the Streamlit live-attack demo's v2 scope is now also built and verified** — a live metrics panel (distance-to-goal, C4 jerk/safety indicators) and a classifier/recovery overlay (online failure-mode classifier's live `p_fail`, Recovery v4's trigger/blend-weight `w`) were added to `app/live_attack_demo.py`, reusing `evaluation/causal_features.py`'s `build_causal_features_online`, `evaluation/episode_runner.py`'s `_pnp_spatial_fields` helper, and `recovery/recovery_v4.py`'s `TriggerWeight`/`get_class_probs` rather than reimplementing any of that logic. The overlay is telemetry-only — it does not wire Recovery v4's blended action into the executed action; the demo still always executes the (possibly attacked) SAC+HER policy action. Verified via real-browser (Playwright) testing: a full auto-play episode to completion, and a `sensor_dropout` run that correctly reproduced the documented near-1.0 `p_fail` / `never_reached_object` finding in real time. Both v2 items from the prior entry's Next Steps are now done — nothing outstanding on this workstream.

**2026-07-21 (later the same day): the Streamlit demo's v3 scope — a dual-pane "Recovery Comparison Demo" — was built, with two real bugs found and fixed.** `app/live_attack_demo.py` was rewritten around a `PaneState` container and a shared `step_pane()` helper; two synchronized `SimWorker` subprocesses now run the same seed/condition in lockstep — raw SAC+HER on the left, SAC+HER + Recovery v4 (CCAR) on the right, with the right pane now genuinely calling `recovery.recovery_v4.recovery_step()` and executing the blended action (previously the v2 overlay was telemetry-only). A "Trustworthiness — this run" section shows both the real per-episode outcome and a lookup into the committed 150-episode `results/data_recovery_v4` benchmark. Two real, root-caused bugs were found during real-browser verification and fixed: (1) `apply_sensor_attack` never threads a seed to `sensor_bias`/`goal_spoof_*`/`object_pose_spoof`'s sampling, so the two panes would get different corruption instances — fixed via `precompute_shared_attack_vectors`/`ensure_shared_attack_vectors`; (2) `env.render()` is not side-effect-free in this MuJoCo/gymnasium_robotics env family — `FetchEnv._render_callback()`'s `mj_forward()` call measurably perturbs the contact solver's warm-start state, meaning live-rendered outcomes could diverge from the headless batch eval purely because the demo renders every step — fixed via `_render_preserving_physics()`'s full `mj_copyData()` snapshot/restore. Committed as `132f702` (v3) and `7f8aebb` (both bug fixes, bundled with an unrelated parallel workstream's commit) — both landed as direct commits between sessions, not staged-and-held (predates CLAUDE.md's git-discipline rule).

**2026-07-22: first-ever multi-session concurrency stress test found and fixed a real crash bug.** The dual-pane demo spawns 2 `SimWorker` subprocesses per browser session — never stress-tested before. A Playwright-driven real-browser test running 4 concurrent sessions reproduced, twice in a row on a clean server, 2-of-4 sessions crashing outright: `multiprocessing`'s spawn-based `Process.start()` reduces a child's `Connection` via a process-global "currently spawning Popen" slot, and since Streamlit runs each session's script in its own thread of one process, concurrent `SimWorker` creation across sessions raced on that global and corrupted which pipe fd wired to which child (`_pickle.UnpicklingError` on the parent's `recv()`, `BrokenPipeError` on the child's `send()`). Fixed in `app/sim_worker.py` with a `threading.Lock` serializing each `SimWorker`'s cold start — narrowing the lock to exclude the handshake wait was tried and measurably made things worse (0/4 sessions completed vs. 4/4 on repeat runs at the wider scope), so the wider scope was kept. `_recv()`'s exception handling was also broadened (`OSError`/`pickle.PickleError`, not just `EOFError`), which exposed and required fixing a second bug in `render_trustworthiness_section` (a crashed pane was marked `done=True`, rendering the trustworthiness comparison as if the episode had legitimately finished). Re-verified with 3 more 4-concurrent-session runs post-fix: 0 crashes (down from a consistent 2/4 crash rate pre-fix). Two more findings from the same stress test were surfaced but NOT fixed (out of scope for this pass): `SimWorker` subprocesses are never cleaned up when a browser tab closes while the server keeps running (no cleanup hook exists), and the server spawns and immediately orphans one throwaway `SimWorker` pair at boot, before any real session connects. Staged, not committed (per CLAUDE.md git discipline).

## Next Steps
**Re-planned 2026-07-22 in direct response to mentor feedback** (see Current Status) —
supersedes the previous plan's "shift to paper writing" priority.

1. **Build the poster/video results package (highest near-term priority, not yet built).**
   Summary tables + charts (success rate by condition/method, CI/significance visualizations)
   + brief interpretive text for `v4_hx`, `v4_hx2`, `v4_hx3`, and the do-no-harm audit.
   Source data already exists and is validated: `results/data_recovery_v4_hx/`,
   `results/data_recovery_v4_hx2/`, `results/data_recovery_v4_hx3/`,
   `results/recovery_do_no_harm_audit.csv`, `results/recovery_v4_hx3_evaluation.csv`. This is
   packaging/presentation work, not new experiments.
2. **Pursue one more "additional recovery improvement"** per the mentor's explicit ask for
   more than the one confirmed win. Recommended next candidate: investigate why `v4_hx`/
   `v4_hx2` underperform `v3` on `goal_spoof_immediate`/`goal_spoof_midep` — real headroom is
   already proven there (`v3` gets +2.9pp/+4.9pp over `sac_her`), unlike the hard-ceiling
   conditions. Calibrating `STAGE_EXPERT_SOFT_WEIGHT` / `LEVEL4_ACTION_ACTUATION_DOWNWEIGHT` /
   `LEVEL4_CONFIDENT_THRESH` is the next-best candidate if that investigation stalls.
3. Recommended sequencing given ~9 days left: front-load item 1 (fast, de-risked, guaranteed
   value regardless of item 2's outcome), then spend remaining time on item 2 -- flagged as a
   recommendation, not a fixed decision; open to running them concurrently if preferred.
4. Lower priority, not urgent since paper writing itself is paused: whether the paper's
   eventual Tier 1 CCAR writeup needs a caveat on `grip_state_falsification` (plain v4's
   confirmed harm there), and whether `hx3`'s null result is worth a one-line paper mention.
5. **Paper writing, JCDL scope-vs-runway, and the Item-1/Level-2-scope paper-framing
   conversations are all explicitly paused per the mentor's direction** — not "open whenever
   wanted" anymore; do not resume until told.
6. Timeline: July 31, 2026 IEEE BigData 2026 REU Symposium deadline (~9 days out as of
   2026-07-22).
7. Demo track (separate from the HX/recovery-integration work above): decide whether to fix
   the two remaining known issues from the 2026-07-22 stress test (tab-close orphan leak,
   boot-time phantom worker pair) — both documented, neither blocking, no urgency established
   yet.

## Open Questions / Blockers
- Two competing trustworthiness formulas (`metrics.py` weights vs. paper's Eq. 2) — still unresolved; the reconciliation script/audit trail didn't survive migration. Not touched in Phase 0.5.
- Sparse-vs-dense online-classifier query cadence (Recovery v4 Phase C) — undecided; gates whether the `reported_grasp` Option B/C alternatives become worth revisiting.
- **Level 2's `clean_2M`-only scope — paper-framing conversation still not had**, flagged across multiple sessions now. Directly relevant to Item 1's hierarchical result (see Next Steps #7): Level 2 is one of the three upstream signals feeding the chain, so its scope limitation and the OOF-stacking-vs-stage-conditioning question should likely be resolved together, not separately.
- **Level 5's `stop_safely` proxy has a documented, unfixed limitation** (see Current Status) — not a blocker on the new recovery-integration plan, since that plan routes around it via Level 1/4 conditioning rather than depending on Level 5's specific rule.
- **Plain `sac_her_recovery_v4`'s confirmed harm on `grip_state_falsification`** (do-no-harm audit, 2026-07-21) is a new, paper-relevant open question — not a blocker on v4_hx2 adoption, but a separate decision about whether/how the existing Tier 1 CCAR section should acknowledge it.
- **Causal flat RF baseline — reference material for Item 1, not a comparison-point substitute (corrected 2026-07-20, see Next Steps item 1).** `causal_feature_matrix.csv` has 14 rows/episode (one per checkpoint_t, same hindsight label repeated); the project's existing pooling convention (Phase 9B `train_causal_classifier.py`, also used for the saved Phase C online classifier) trains/evaluates on all 14 checkpoints pooled. This pooled RF is checkpoint-level, not episode-level, so it characterizes detection-delay/false-alarm behavior usefully but does not substitute for the episode-level flat-vs-hierarchical comparison memo Section 4 calls for. Detection-delay characterization (2026-07-19, `retrain/causal-flat-rf-baseline`) confirms the pooled number is optimistic *early* in the episode specifically (t=19–59: acc 0.917→0.935, macro-F1 0.765→0.801; false-alarm rate on clean+success rows ~70–77%) and stabilizes from t≈69 onward (acc ~0.955–0.961, macro-F1 ~0.855–0.881, false-alarm rate ~7–10%). Remaining open question is narrower than before and not blocking Item 1: whether a live recovery trigger should query the model at every step (as trained) or wait past the ~t=69 inflection — that's an operating-point decision deliberately left unmade pending review, relevant to recovery-trigger design more than to the flat-vs-hierarchical comparison itself.
- **False-alarm early→mid drop — now statistically confirmed (2026-07-20).** The steep drop noted above (n=420, ~30/checkpoint) was too sparse per-checkpoint to cite with confidence. Bootstrap CIs (95%, 5000 resamples) on the same data: per-checkpoint CIs are wide (n=30 each, e.g. t=19: 0.733 [0.567, 0.867]) but the early/mid/late-binned CIs are not — early (t=19–59, n=150): 0.720 [0.647, 0.793]; mid (t=69–109, n=150): 0.067 [0.027, 0.107]; late (t=119–149, n=120): 0.100 [0.050, 0.158]. Early-vs-mid CIs do not overlap (gap=0.540) — the drop is real, not noise. Mid-vs-late CIs overlap substantially — no evidence of further change after t≈69. This is the empirical basis cited for Phase 1 Item 2 (task-stage recognition); it is evidence only, not a cadence or operating-point decision (still open, see above). Script: `scripts/bootstrap_false_alarm_ci.py`. Output: `results/classifier_causal_baseline/false_alarm_ci.txt`.
- **Demo's remaining concurrency findings are documented but unfixed** (see Current Status 2026-07-22): orphaned `SimWorker` subprocesses on tab-close, and a boot-time phantom pair. Not a blocker on anything else; flagged so a future session doesn't have to rediscover them.

## Key Files & Environment
- Repo: `/Users/yves/Documents/Github/NSF-REU-TAIRO` (conda env `reu_robotics`, Python 3.11, stable_baselines3 2.8.0). Use `/opt/miniconda3/envs/reu_robotics/bin/python3` directly — `conda run -n reu_robotics` silently falls back to system Python in this shell.
- How to run: always activate/use the `reu_robotics` env (see CLAUDE.md §15). Canonical paper data lives in `results/data_seedfix/`, not `results/data/` (doesn't exist in this repo).
- Important paths/configs: `config.py` (constants source of truth); `CLAUDE.md`, `findings.md`, `RECOVERY_V4.md`, `ATTACK_AWARE_TRACK.md` (all gitignored — not versioned); `reported_grasp_contact_options.md` (tracked).
- Streamlit live demo: `app/live_attack_demo.py` (UI/session-state/control flow) + `app/sim_worker.py` (subprocess-isolated MuJoCo rendering — required on macOS; see Session Log 2026-07-21).

## Session Log

### 2026-07-22 (Streamlit demo: multi-session stress test + concurrency crash fix)
- **What changed:** Picked up the Streamlit demo workstream from `session_handoff_7.md`.
  Ran the previously-flagged, never-done multi-session resource stress test: started
  `app/live_attack_demo.py`, used Playwright to drive 4 concurrent real-browser sessions
  (distinct condition/seed each) through full autoplay episodes, and monitored
  memory/CPU/process counts via `ps` sampling (no new deps added). Found and fixed a real
  crash bug, then synced this file.
- **Decisions made:** (1) Investigate and attempt a fix for the crash bug rather than just
  document it — confirmed via sign-off after reporting the initial finding, given its
  severity (a guaranteed ~50% per-session crash rate under concurrent load). (2) Keep the
  `_SPAWN_LOCK` scope wide (covering the initial handshake `recv()`, not just
  `Pipe()`+`Process.start()`) after empirically testing the narrower scope and finding it
  performed worse, not better.
- **Rationale / results:** 4 concurrent sessions reproducibly crashed 2-of-4 outright on a
  clean server (`_pickle.UnpicklingError` on the parent's `recv()`, `BrokenPipeError` on the
  child's `send()`) — root-caused to `multiprocessing`'s spawn-based `Process.start()` racing
  on a process-global "currently spawning Popen" slot when called concurrently from different
  threads of the same process (Streamlit runs each session in its own thread). Fixed with a
  `threading.Lock` around each `SimWorker`'s cold start in `app/sim_worker.py`, plus a
  broadened `_recv()` exception clause (`OSError`/`pickle.PickleError`, not just `EOFError`)
  so any remaining IPC failure degrades to the app's own graceful `SimWorkerCrashed` state
  instead of a raw Streamlit traceback. That graceful-degradation fix exposed a second,
  independent bug — `render_trustworthiness_section` treated a crashed pane (`done=True` but
  `success=None`) as a legitimately finished episode — fixed by checking `worker_error` first.
  Re-verified with 3 more 4-concurrent-session runs on the final code: 0 crashes each time
  (previously a consistent 2/4 crash rate). Two more findings were surfaced but deliberately
  left unfixed this session (documented in Current Status/Open Questions instead): orphaned
  `SimWorker` subprocesses on tab-close (no cleanup hook exists at all), and a boot-time
  phantom `SimWorker` pair that spawns before any real session connects. Cross-session UI
  isolation itself was confirmed correct throughout — every session (including ones that
  crashed) always showed its own condition/seed, never another session's.
- **Blockers hit:** None on the stress test/fix itself. Discovered mid-session that a
  concurrent process had staged substantial unrelated changes to this very file (the `hx3`
  evaluation + mentor-feedback re-plan, entry above) between when this session first read it
  and when it went to write — re-read the current staged content before editing so this
  entry layers on top of that work rather than clobbering it. `app/sim_worker.py` and
  `app/live_attack_demo.py` were staged (not committed) per CLAUDE.md git discipline; other
  concurrently-staged files (from the other track) were left untouched.

### 2026-07-22 (hx3 evaluated + mentor feedback re-plan)
- **What changed:** Implemented and evaluated `sac_her_recovery_v4_hx3` (new file
  `recovery/recovery_v4_hx3.py`, wired into `config.py`/`evaluation/episode_runner.py`/
  `scripts/run_multiseed_sweep.py`) -- re-gates `relocalization_expert` on
  `max(class_probs["spoofed_goal"], l4_probs["perception_state"])` instead of the near-inert
  `spoofed_goal`-only signal. Evaluated immediately at full power (n=450, seeds 0-14, all 11
  conditions, `results/data_recovery_v4_hx3/`) via new `scripts/evaluate_recovery_v4_hx3.py`
  (reuses `scripts/audit_recovery_do_no_harm.py`'s stats helpers, no duplicated code). Then
  received mentor feedback (quoted in Current Status) that reversed the "shift to paper
  writing" plan from the prior entry -- re-planned Next Steps in response.
- **Decisions made:** (1) Combine `spoofed_goal`/`perception_state` via plain `max()`, not a
  weighted blend -- confirmed via sign-off, to avoid a 4th uncalibrated constant on top of the
  existing three. (2) Evaluate at full n=450 immediately rather than a smaller pilot first,
  per this session's own "n=150 underpowered" lesson. (3) After mentor feedback arrived,
  re-planned toward two parallel streams -- packaging existing results for poster/video
  (tables/charts/stats/interpretations) and one more recovery-improvement attempt
  (goal-spoof underperformance vs. `v3`) -- rather than treating the `hx3` null result as a
  stopping point.
- **Results:** `hx3` produced a genuine, well-powered null on its target
  (`object_pose_spoof`: -2.2pp vs. `v4`, +0.9pp vs. `v4_hx2`, neither significant after BH
  correction), did no new harm anywhere (do-no-harm check clean across all 11 conditions), and
  left `v4_hx2`'s `grip_state_falsification` win intact. **`v4_hx2` remains the sole adopted
  variant.** Full detail: `RECOVERY_V4.md` §5.9.
- **Blockers hit:** None new this entry (the concurrent Streamlit-track session's repeated
  direct commits, flagged in the prior entry, continued but were left untouched per the user's
  standing choice -- not re-investigated further).

### 2026-07-21 (recovery-integration power check + do-no-harm audit + fold-in decision)
- **What changed:** Extended the recovery-variant power check (n=450, seeds 0-14) from the
  previously-checked 4 conditions to the remaining 7 (`sensor_bias`, `sensor_dropout`,
  `action_clipping`, `action_reversal`, `contact_dropout`, `goal_spoof_immediate`, `clean`),
  completing the full 11-condition grid for `sac_her_recovery_v4`/`_hx`/`_hx2`. Backfilled
  `sac_her` onto seeds 5-14 across all 11 conditions so every recovery-vs-baseline comparison
  could use a paired test. Built a new systematic do-no-harm audit
  (`scripts/audit_recovery_do_no_harm.py`) comparing all 5 recovery methods (v2/v3/v4/v4_hx/v4_hx2)
  against plain `sac_her`, 55 comparisons, Benjamini-Hochberg FDR corrected. Updated
  `RECOVERY_V4.md` (§5.3 update, new §5.6/§5.7) and `findings.md` (Phase 11 update) with the
  full-grid results.
- **Decisions made:** (1) Backfill `sac_her` onto seeds 5-14 rather than use unpaired tests,
  so every comparison in the audit is a paired McNemar test — confirmed via sign-off, ~30 min
  extra compute for consistent methodology across the whole audit. (2) Apply BH-FDR correction
  given this audit is a systematic 55-comparison scan (unlike RECOVERY_V4.md §5.3's original
  2-condition targeted test, which didn't need one) — confirmed via sign-off. Applied the same
  correction retroactively to the original v4_hx/v4_hx2-vs-v4 comparison once extended to the
  full 11-condition grid (22 comparisons), which changed one conclusion (see Rationale).
- **Rationale / results:** The do-no-harm audit found that plain `sac_her_recovery_v4` — already
  the paper's headline B4 method — does statistically significant harm on
  `grip_state_falsification` relative to no recovery at all (19.8%→14.0%, BH-adjusted
  p=0.00017); `v4_hx` inherits this unchanged; `v4_hx2` corrects it back to statistical parity
  with doing nothing (18.2%, not significantly different from 19.8%). No other method/condition
  pair in the 55-comparison grid reached significance either direction. Separately, extending
  the original v4-variant-vs-v4 comparison to the full 11-condition grid changed one earlier
  conclusion: the `object_pose_spoof` regression previously reported as a "real, significant
  harm" for `v4_hx` (session_handoff_5, based on a 4-condition test) no longer reaches
  significance once BH-corrected across the full 22-comparison grid (BH-adjusted p=0.180) —
  same point estimate and mechanism, downgraded from confirmed to plausible-but-unconfirmed.
  The `grip_state_falsification` win for `v4_hx2` (+4.2pp) holds at full-grid correction.
- **Blockers hit / notable events:** Mid-session, discovered a separate, apparently-concurrent
  Claude Code session had been working on an unrelated Streamlit demo track (`app/`) on this
  same repo throughout the day, committing directly to `main` three times (`71e03bc`, `4f38b6e`,
  `132f702`) rather than staging — the third commit (20:25) postdates `session_handoff_6.md`
  and appears to add recovery-blending to the demo, a feature that handoff explicitly flagged
  as needing the user's sign-off first, with no recorded sign-off. Flagged to the user, who
  chose to leave it untouched and review separately — not investigated further here, no files
  from that track touched. Also confirmed at session start that the *previous* session's own
  work (session_handoff_5, Level 5 + v4-HX/HX2 build) had landed as a direct commit
  (`45c8c51`) rather than staged, contrary to that handoff's own claim — user chose to proceed
  without further action on it.
- **Follow-up decision (same session):** after being shown the do-no-harm audit results in
  plain language and the two linked open questions (fold `v4_hx2` into the paper? caveat plain
  v4?), the user directly confirmed **`sac_her_recovery_v4_hx2` folds into the paper's Recovery
  v4 section** — recorded in `RECOVERY_V4.md` §5.7 and Next Steps above. The plain-v4-caveat
  question was not addressed in this exchange and remains open.

### 2026-07-21 (second entry)
- **What changed:** Added the two v2 features to `app/live_attack_demo.py`: a live metrics panel (distance-to-goal via `distance_to_goal`, C4 jerk/safety via the same per-channel split-jerk formula as `episode_runner.py`) and a classifier/recovery overlay (online failure-mode classifier's `p_fail`, EMA(`p_fail`), and Recovery v4's blend weight `w`). New session-state fields track `prev_executed` (jerk comparand) and a growing `step_history` list (the step-log schema `build_causal_features_online` needs), plus a per-episode `TriggerWeight` instance recreated on Reset.
- **Decisions made:** (1) Reused `evaluation/episode_runner.py`'s `_pnp_spatial_fields` and `evaluation/causal_features.py`'s `build_causal_features_online` directly rather than reimplementing the spatial-feature/causal-feature math — confirmed these are the same functions the real `sac_her_recovery_v4` sweep path uses, verified their exact call order/inputs against `recovery_v4.py`'s `recovery_step` before wiring. (2) The overlay is explicitly display-only — `w` and the classifier's class probabilities are computed and shown but never blended into `executed_action`, keeping this addition strictly additive to v1's control flow. (3) Classifier artifacts (`results/classifier_seedfix/online_failure_classifier.pkl`, `recovery_v4_trigger_calibration.pkl`) are loaded once via a second `@st.cache_resource` function, mirroring `load_policy()`'s existing pattern.
- **Rationale:** Given how much v1 debugging turned on subtle real-browser-only behavior, the same validation discipline was applied here: logic was first checked standalone (a bare Python loop calling the same functions, confirming causal features and classifier calls work end-to-end at ~12ms/step) before touching Streamlit, then the full app was checked in a real Playwright-driven Chromium session, not just `AppTest`. A `sensor_dropout` run was used as a real-world sanity check — since CLAUDE.md documents that condition's near-3% success rate as a headline finding, the classifier overlay correctly holding `p_fail≈1.0` / `never_reached_object` throughout that run is independent confirmation the wiring is correct, not just crash-free.
- **Blockers hit:** None. Both v2 items are complete; no further Streamlit-demo work is currently planned.

### 2026-07-21
- **What changed:** Built a new interactive Streamlit demo (`app/live_attack_demo.py`, `app/sim_worker.py`) running the SAC+HER `clean_2M` PickAndPlace policy live against user-selected attack conditions, with a real-time MuJoCo render pane and Start/Pause/Step/Reset playback controls. Added `streamlit` to `requirements.txt`. Committed as `71e03bc streamlit implementation v1`.
- **Decisions made:** (1) MuJoCo's rgb_array rendering runs in a dedicated subprocess (`SimWorker` in `sim_worker.py`), not in-process — confirmed via direct reproduction that GLFW's OpenGL context creation is main-thread-only on macOS, and Streamlit always executes the app script in a background ScriptRunner thread, so in-process rendering hard-crashes (SIGTRAP, uncatchable). (2) Render resolution bumped to 960x960 with a closer/lower camera (chosen by visually comparing candidate frames) — cosmetic only, scoped to `sim_worker.py`, doesn't touch the shared `envs/fetchpickandplace_env.py` used by training/eval/recording. (3) The render+step loop lives in a `st.fragment` so auto-play reruns don't flicker/scroll-reset the whole page. (4) The fragment's `run_every` poll rate is fixed at 0.25s (4Hz), with the FPS slider capped to 1-4 to match — empirically, anything faster than ~0.1s races against normal button-triggered full-page reruns and permanently blanks the render pane (confirmed reproducible at 0.05s, confirmed fine at 0.1s-3.0s, via a real Playwright-driven Chromium instance, not Streamlit's own AppTest harness which doesn't exercise this).
- **Rationale:** Two separate real bugs were found and fixed via direct browser-level reproduction rather than guesswork: (a) the macOS GLFW main-thread crash, root-caused by reproducing it with a bare `threading.Thread` with no Streamlit involved at all; (b) the auto-play "blank pane" bug, root-caused by installing Playwright/Chromium mid-session specifically to get real DOM/network-level evidence after AppTest-based testing kept passing cleanly (AppTest doesn't simulate the browser-side timer that drives `run_every`, so it couldn't catch either issue). Also flagged and worked around: VS Code's embedded browser/webview doesn't reliably support the persistent WebSocket Streamlit's interactivity needs — page loads but clicks silently do nothing; fix is to use a real external browser.
- **Blockers hit:** None outstanding — v1 is working and verified end-to-end (manual stepping, condition switching, auto-play, Reset, episode completion/success banner) via a real browser. v2 scope (metrics panel, classifier/recovery overlay) explicitly deferred, now captured in Next Steps.

### 2026-07-20 (second entry)
- **What changed:** Saved Item 1's hierarchical RF/XGBoost as `.pkl` artifacts (`results/classifier_hierarchical/`). Built Level 5 (`scripts/build_level5_labels.py` → `results/level5_decisions.csv`), evaluated it against real episode outcomes (`scripts/evaluate_level5_decisions.py` → `results/level5_outcome_eval.csv`), and attempted threshold recalibration (`scripts/calibrate_level5_stop_safely.py`, diagnostic only, no files written). Re-raised and reconfirmed the Level 3 multi-label deferral. Had the deferred paper-framing and JCDL scope-vs-runway conversations.
- **Decisions made:** Level 5 consumes chained Levels 2-4 predictions (not ground truth, not Level 1), uses the memo's 7-decision taxonomy, and stays fully separate from `recovery_v4.py` — all confirmed via sign-off before implementation. After the outcome evaluation surfaced `stop_safely`'s false-fire problem, recalibration was attempted and found ineffective (not a threshold issue). Given that finding, the user's mentor redirected the project: the classifier hierarchy should actively improve recovery this week, not remain a separate track. Agreed plan: stage-gated expert mixture in `recovery_v4.py` using Level 1, shipped as a new A/B-able method variant, evaluated on the existing clean_2M grid, with Level 4 as a possible follow-on refinement.
- **Rationale:** Level 5's rule-based safe-stop trigger fails not because of a bad constant but because checkpoint-granularity (14/episode) genuinely doesn't carry the information to separate recoverable from unrecoverable trajectories — recoverable episodes look sustained-abnormal for almost their whole duration until recovery_v4 saves them very late. That finding redirected effort toward a different, better-motivated integration path (stage-gating, matching the memo's own core example) rather than continuing to tune an approach with a structural ceiling.
- **Blockers hit:** None new — the paper-framing conversation remains explicitly deferred (unchanged), and Level 3 multi-label remains deferred (unchanged, cost estimate confirmed still valid).

### 2026-07-20
- **What changed:** Built Level 1's classifier (`scripts/train_level1_classifier.py`, mirroring `scripts/train_level4_classifier.py`'s template), then built the out-of-fold chaining plumbing needed to turn Levels 1/2/3/4 from independent standalone fits into an actual hierarchical pipeline: `evaluation/oof_chaining.py` (generic leave-one-seed-out OOF utility), `scripts/build_hierarchical_chain.py` (runs the Level 1→2→3→4 chain), `scripts/train_hierarchical_classifier.py` (episode-level hierarchical RF/XGBoost). Ran the full pipeline end-to-end, closing Item 1's four-way comparison.
- **Decisions made:** (1) Level 1's 100% accuracy reported honestly rather than forcing a harder test by stripping the cascade-defining features — confirmed via sign-off, since Level 1's hierarchy role is reliable online stage surfacing, not indirect-signal generalization. (2) OOF strategy = leave-one-seed-out across the existing `TRAIN_SEEDS=[0,1,2,3]` convention (4-fold, each holding out one seed), not a generic random K-fold, to match repo precedent. (3) Chain conditioning scope = immediate-predecessor only (1→2, 2→3, 3→4), matching CLAUDE.md's singular "the upstream level" wording, not cumulative. (4) Item 1's episode-level rollup = each episode's final checkpoint (t=149)'s chained prediction, mirroring how the flat baselines are themselves evaluated on full-episode features. (5) Level 2's source model and Level 3's model both predict `failure_mode` (Level 2's p_fail was always derived that way, confirmed by reading `scripts/build_online_classifier.py`) — not circular, since Level 3 only ever sees Level 2's OOF-safe output, never in-sample.
- **Rationale:** Naively calling an upstream model's `.predict()` on its own training rows to build a downstream level's features would leak in-sample confidence (memorized rows would look artificially easy), so genuine OOF predictions were the prerequisite for any real chaining. All OOF-strategy/chain-scope/rollup decisions were confirmed with the user before implementation (three judgment calls with no prior repo precedent) rather than assumed silently, per this project's established norm.
- **Blockers hit / findings surfaced, not resolved:** `PROJECT_CONTEXT.md` (this file) and `session_handoff_3.md` (Claude-Code-only) had diverged on Item 1's granularity requirement before this session reconciled them — `PROJECT_CONTEXT.md`'s 2026-07-20 correction (episode-level required) predates this session's write-up but wasn't yet acted on when Levels 1/4's checkpoint-level classifiers were built earlier the same day; this session's episode-level rollup step resolves that gap. The hierarchical result's dominant driver is an OOF-stacking effect (Level 3's own `failure_mode` OOF prediction as a meta-feature) rather than purely stage-conditioning reinterpretation — flagged, not resolved, feeds directly into the paper-framing conversation (see Open Questions / Blockers).

### 2026-07-18
- **What changed:** Full TAIRO-HX Phase 0 (read-only migration audit) and Phase 0.5 (open-item resolution) completed across several sessions on branch `investigate/ws2-metric-and-grasp-def`. Fresh-retrained the failure-mode classifier against `results/classifier_seedfix/feature_matrix.csv` to verify the documented Phase 8.5 metric; ran a quantified causal-vs-hindsight grasp-disagreement analysis with a dense-vs-sparse cadence comparison. Branch merged to `main` (fast-forward, `b63c632..be19867`) and deleted. Commit message for the grasp-proxy resolution was amended for accuracy. CLAUDE.md §10's stale auto-load claim corrected. Repo pushed to origin.
- **Decisions made:** WS2 headline metric updated to 0.9424/0.8159 (current, reproducible) in place of the stale 0.996/0.779. `reported_grasp`/`contact` causal proxy: Decision Option A (kept as-is, documented, not changed). New project-level decision-making norm adopted: taxonomy/definitional calls are now made directly by the team, documented with evidence, rather than gated on external review turnaround.
- **Rationale:** The seed-independence fix (a known correctness fix) is what changed the underlying episode population between the old and new WS2 numbers — old accuracy was inflated by duplicate-episode leakage, not a stronger classifier. Option A chosen because the causal proxy's disagreement with hindsight is small and strictly conservative (never a false-positive grasp claim), and the alternative (lift-trend check) only pays off at a query cadence (dense) the project hasn't adopted yet. This decision-making norm change reflects a move toward faster, self-documented resolution of taxonomy/definitional questions as the project approaches its submission deadline.
- **Blockers hit:** Trustworthiness-formula reconciliation still open (script/audit trail lost in migration). Sparse-vs-dense classifier cadence decision still pending, gating further `reported_grasp` work.
