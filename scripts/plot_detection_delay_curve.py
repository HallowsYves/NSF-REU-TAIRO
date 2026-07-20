"""
Retrain prep — Item 1 detection-delay evaluation, Task 3.

Plots accuracy and macro-F1 of the pooled causal classifier
(results/classifier_causal_baseline/pooled_model.pkl) across its 14
checkpoints, from the curve data written by
scripts/train_and_save_causal_pooled.py.

Observation-only: this script does not select or mark an "operating point"
checkpoint — that is a decision for review, not something to bake into the
plot.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = "results/classifier_causal_baseline"
CURVE_CSV = os.path.join(OUT_DIR, "_checkpoint_curve_data.csv")
OUT_PNG = os.path.join(OUT_DIR, "detection_delay_curve.png")

df = pd.read_csv(CURVE_CSV).sort_values("checkpoint_t")

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_ACC = "#2a78d6"   # categorical slot 1 (blue)
SERIES_F1 = "#008300"    # categorical slot 2 (green)

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(df["checkpoint_t"], df["accuracy"], color=SERIES_ACC, linewidth=2,
        marker="o", markersize=5, label="Accuracy")
ax.plot(df["checkpoint_t"], df["macro_f1"], color=SERIES_F1, linewidth=2,
        marker="o", markersize=5, label="Macro F1")

ax.set_xlabel("Checkpoint timestep (steps into episode)", color=TEXT_PRIMARY)
ax.set_ylabel("Score", color=TEXT_PRIMARY)
ax.set_title("Causal classifier detection delay — pooled model, seed-4 test",
             color=TEXT_PRIMARY, fontsize=12)
ax.set_ylim(0.7, 1.0)
ax.set_xticks(df["checkpoint_t"])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

ax.grid(True, color="#dddddb", linewidth=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cccccb")
ax.tick_params(colors=TEXT_SECONDARY)

legend = ax.legend(loc="lower right", frameon=False, labelcolor=TEXT_PRIMARY)

fig.tight_layout()
fig.savefig(OUT_PNG, facecolor=SURFACE)
print(f"[plot] Saved -> {OUT_PNG}")
