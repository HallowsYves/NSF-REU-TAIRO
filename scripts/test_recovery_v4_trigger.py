"""
Recovery v4 Tier 1 -- isolated sanity test for TriggerWeight (Phase 1).

Not a pytest suite; a standalone script run once to confirm the trigger
weight rises and falls sensibly before wiring it to the classifier or the
episode runner. Uses the calibrated clean_2M midpoint (the one checkpoint
with a non-degenerate clean baseline; see Phase 1 report) plus a couple of
synthetic edge cases.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recovery.recovery_v4 import TriggerWeight

CLASSIFIER_DIR = "results/classifier_seedfix"

with open(os.path.join(CLASSIFIER_DIR, "recovery_v4_trigger_calibration.pkl"), "rb") as f:
    calibration = pickle.load(f)


def run_sequence(label, p_success_seq, midpoint, alpha=0.1):
    trigger = TriggerWeight(clean_pfail_p95=midpoint, alpha=alpha)
    ws = []
    ema_history = []
    for p in p_success_seq:
        w = trigger.update(p)
        ws.append(w)
        ema_history.append(trigger.ema_pfail)  # snapshot AFTER this step's update
    print(f"\n{label}  (midpoint={midpoint:.4f}, alpha={alpha})")
    for t, (p, ema, w) in enumerate(zip(p_success_seq, ema_history, ws)):
        print(f"  t={t:>3}  p_success={p:.2f}  ema_pfail={ema:.4f}  w={w:.4f}")
    return ws


# -- Case 1: step function, clean_2M calibrated midpoint --------------------
# 20 steps healthy (p_success=0.9), then a 20-step failure run (p_success=0.1).
seq = [0.9] * 20 + [0.1] * 20
ws = run_sequence("Case 1: step 0.9 -> 0.1, clean_2M midpoint",
                   seq, calibration["clean_2M"])

w_before = max(ws[:20])
w_after = max(ws[20:])
assert w_after > w_before, (
    f"FAIL: w did not rise after sustained failure (before={w_before:.4f}, after={w_after:.4f})"
)
print(f"\n  -> w rose from <= {w_before:.4f} (healthy) to {w_after:.4f} (failing): PASS")

# -- Case 2: recovery -- failure then sustained success ---------------------
seq2 = [0.1] * 20 + [0.9] * 30
ws2 = run_sequence("Case 2: step 0.1 -> 0.9 (recovery), clean_2M midpoint",
                    seq2, calibration["clean_2M"])
w_fail = max(ws2[:20])
w_recovered = ws2[-1]
assert w_recovered < w_fail, (
    f"FAIL: w did not fall after sustained success (fail={w_fail:.4f}, recovered={w_recovered:.4f})"
)
print(f"\n  -> w fell from {w_fail:.4f} (failing) to {w_recovered:.4f} (recovered): PASS")

# -- Case 3: always-clean sequence should keep w near 0 for clean_2M --------
seq3 = [0.95] * 50
ws3 = run_sequence("Case 3: sustained p_success=0.95, clean_2M midpoint",
                    seq3, calibration["clean_2M"])
assert max(ws3) < 0.05, (
    f"FAIL: w rose above the 0.05 clean-episode target (max w={max(ws3):.4f})"
)
print(f"\n  -> w stayed below 0.05 target throughout ({max(ws3):.4f} max): PASS")

# -- Case 4: degenerate midpoint (p95=1.0, e.g. clean_500k/randomized_*) ----
# Flagged separately in the Phase 1 report -- included here to document the
# actual (degenerate) behavior rather than hide it.
seq4 = [0.9] * 20 + [0.1] * 20
ws4 = run_sequence("Case 4: step 0.9 -> 0.1, DEGENERATE midpoint=1.0 (clean_500k)",
                    seq4, calibration["clean_500k"])
print(f"\n  -> max w over full sequence: {max(ws4):.4f} (see Phase 1 report: midpoint=1.0 "
      f"means the sigmoid never reaches its transition region for any p_fail <= 1.0)")

print("\nAll TriggerWeight sanity checks passed (cases 1-3).")
