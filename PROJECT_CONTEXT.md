# TAIRO (NSF-REU-TAIRO) — Context

## Overview
- Goal: TAIRO evaluates SAC+HER robot manipulation policies under 11 adversarial attack conditions, producing a five-component trustworthiness score (C1–C5). Currently scoping TAIRO-HX, a proposed hierarchical failure-diagnosis redesign (Task Stage → Anomaly Detection → Failure Type → Attack Family → Recovery Decision).
- Constraints or deadlines: IEEE BigData 2026 REU Symposium submission target.

## Current Status
TAIRO-HX Phase 0/0.5 pre-implementation audit is complete; all open items it surfaced are resolved. Phase 0 (read-only migration audit) found WS1/WS3 intact and WS4 intact in code. Phase 0.5 closed the two remaining substantive items: (1) WS2 classifier metric mismatch — root cause confirmed as pre-seed-independence-fix duplicate-episode leakage inflating the old 0.996/0.779 figure; that intermediate dataset didn't survive migration and isn't recoverable. Current truth (0.9424 accuracy / 0.8159 macro-F1) is fully reproducible from `results/classifier_seedfix/` via fresh `.fit()`; CLAUDE.md and findings.md corrected accordingly. (2) `reported_grasp`/`contact` causal-vs-hindsight definitional gap — quantified (4.61% episode disagreement, 304/6,600, always one-directional/conservative, never a false-positive grasp claim) and closed via Decision: Option A, keep the causal proxy as-is; documented in-code (`causal_features.py`) and in `reported_grasp_contact_options.md`. The smoke-test-pruning question from Phase 0 was closed as informational-only (confirmed exploratory, not load-bearing). CLAUDE.md §10's stale claim about automatic seedfix classifier loading was also corrected this session. All work happened on `investigate/ws2-metric-and-grasp-def`, fast-forward merged into `main` (`b63c632..be19867`) and deleted; `main` is pushed and in sync with `origin/main` (`be19867`, verified via `git status`/`git rev-parse`). TAIRO-HX Phase 1 implementation has not started.

## Next Steps
1. Flat vs. hierarchical classifier comparison (RF + XGBoost, baseline = 0.9424 / 0.8159).
2. Task-stage recognition + recoverability decision output.
3. Trusted-goal storage.
4. Gated single-family recovery with safe-stop.

## Open Questions / Blockers
- Two competing trustworthiness formulas (`metrics.py` weights vs. paper's Eq. 2) — still unresolved; the reconciliation script/audit trail didn't survive migration. Not touched in Phase 0.5.
- Sparse-vs-dense online-classifier query cadence (Recovery v4 Phase C) — undecided; gates whether the `reported_grasp` Option B/C alternatives become worth revisiting.

## Key Files & Environment
- Repo: `/Users/yves/Documents/Github/NSF-REU-TAIRO` (conda env `reu_robotics`, Python 3.11, stable_baselines3 2.8.0). Use `/opt/miniconda3/envs/reu_robotics/bin/python3` directly — `conda run -n reu_robotics` silently falls back to system Python in this shell.
- How to run: always activate/use the `reu_robotics` env (see CLAUDE.md §15). Canonical paper data lives in `results/data_seedfix/`, not `results/data/` (doesn't exist in this repo).
- Important paths/configs: `config.py` (constants source of truth); `CLAUDE.md`, `findings.md`, `RECOVERY_V4.md`, `ATTACK_AWARE_TRACK.md` (all gitignored — not versioned); `reported_grasp_contact_options.md` (tracked).

## Session Log

### 2026-07-18
- **What changed:** Full TAIRO-HX Phase 0 (read-only migration audit) and Phase 0.5 (open-item resolution) completed across several sessions on branch `investigate/ws2-metric-and-grasp-def`. Fresh-retrained the failure-mode classifier against `results/classifier_seedfix/feature_matrix.csv` to verify the documented Phase 8.5 metric; ran a quantified causal-vs-hindsight grasp-disagreement analysis with a dense-vs-sparse cadence comparison. Branch merged to `main` (fast-forward, `b63c632..be19867`) and deleted. Commit message for the grasp-proxy resolution was amended for accuracy. CLAUDE.md §10's stale auto-load claim corrected. Repo pushed to origin.
- **Decisions made:** WS2 headline metric updated to 0.9424/0.8159 (current, reproducible) in place of the stale 0.996/0.779. `reported_grasp`/`contact` causal proxy: Decision Option A (kept as-is, documented, not changed). New project-level decision-making norm adopted: taxonomy/definitional calls are now made directly by the team, documented with evidence, rather than gated on external review turnaround.
- **Rationale:** The seed-independence fix (a known correctness fix) is what changed the underlying episode population between the old and new WS2 numbers — old accuracy was inflated by duplicate-episode leakage, not a stronger classifier. Option A chosen because the causal proxy's disagreement with hindsight is small and strictly conservative (never a false-positive grasp claim), and the alternative (lift-trend check) only pays off at a query cadence (dense) the project hasn't adopted yet. This decision-making norm change reflects a move toward faster, self-documented resolution of taxonomy/definitional questions as the project approaches its submission deadline.
- **Blockers hit:** Trustworthiness-formula reconciliation still open (script/audit trail lost in migration). Sparse-vs-dense classifier cadence decision still pending, gating further `reported_grasp` work.
