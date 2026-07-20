"""
TAIRO-HX hierarchical level-chaining — generic out-of-fold (OOF) prediction
utility.

Per CLAUDE.md's "Level-Chaining Architecture" decision, each level in the
Level 1->2->3->4 pipeline is a separate model that receives the level
immediately upstream's predicted class (one-hot) + one confidence scalar
(max class probability) as additional input features -- never the full
probability vector. Feeding a downstream level predictions from an upstream
model that was FIT on those same rows would leak in-sample confidence
(rows the upstream model memorized would look artificially easy), so every
upstream signal consumed downstream must be out-of-fold.

OOF strategy (confirmed 2026-07-20): leave-one-seed-out across the existing
TRAIN_SEEDS=[0,1,2,3] convention used everywhere else in this repo (Level
1/4, Phase 8.5/9B/9C) -- 4 folds, each holding out one seed, fit on the
other three. This gives genuinely out-of-fold predictions for every
training row. A final refit on all of TRAIN_SEEDS then produces predictions
for TEST_SEED (already naturally out-of-fold, since it was never trained
on by any fold or the final refit).

episode_idx is globally unique across seeds (verified: seed N's episodes
are offset by N * 990, not reset per seed), so per-fold and train/test
episode_idx-overlap checks are meaningful leakage checks, not false
positives from episode_idx numbering restarting each seed.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

KEY_COLS = ["model", "condition", "seed", "episode_idx", "checkpoint_t"]

DEFAULT_RF_KWARGS = dict(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)


def get_oof_predictions(df, feature_cols, label_col, train_seeds, test_seed,
                         rf_kwargs=None, prefix=None):
    """
    Leave-one-seed-out OOF predictions for `train_seeds` rows, plus a final
    seeds-`train_seeds`-refit prediction for `test_seed` rows.

    Returns
    -------
    preds : pd.DataFrame
        KEY_COLS + f"{prefix}_pred_class" + f"{prefix}_confidence", one row
        per input row (train_seeds union test_seed rows of `df`).
    final_model : fitted RandomForestClassifier
        The refit-on-all-train_seeds model (available for reuse/inspection,
        e.g. to score held-out conditions later).
    proba : pd.DataFrame
        KEY_COLS + one column per class in `classes_sorted` (all OOF/refit,
        never in-sample) -- for callers that need a specific class's
        probability (e.g. Level 2's p_fail = 1 - proba["success"]), not just
        the argmax. NEVER pass this full-vector frame into a downstream
        level's feature set directly -- only `onehot_upstream()`'s
        class+confidence schema is allowed downstream, per CLAUDE.md's
        chaining decision.
    """
    rf_kwargs = rf_kwargs or DEFAULT_RF_KWARGS
    prefix = prefix or label_col

    scoped = df[df["seed"].isin(train_seeds + [test_seed])]
    labeled = scoped.dropna(subset=[label_col])
    classes_sorted = sorted(labeled[label_col].unique())

    fold_frames = []
    proba_frames = []

    print(f"[oof:{prefix}] Leave-one-seed-out OOF over train_seeds={train_seeds} "
          f"(classes={classes_sorted})")
    for held_out in train_seeds:
        fold_train_seeds = [s for s in train_seeds if s != held_out]
        fold_train = labeled[labeled["seed"].isin(fold_train_seeds)]
        fold_test = labeled[labeled["seed"] == held_out]

        overlap = set(fold_train["episode_idx"]) & set(fold_test["episode_idx"])
        assert len(overlap) == 0, \
            f"LEAKAGE: fold held_out={held_out} has {len(overlap)} episode_idx overlap"

        clf = RandomForestClassifier(**rf_kwargs)
        clf.fit(fold_train[feature_cols].values, fold_train[label_col].values)
        proba = clf.predict_proba(fold_test[feature_cols].values)
        proba_df = pd.DataFrame(proba, columns=clf.classes_, index=fold_test.index)
        proba_df = proba_df.reindex(columns=classes_sorted, fill_value=0.0)

        fold_frames.append(_assemble(fold_test, proba_df, prefix))
        proba_frames.append(_with_key(fold_test, proba_df))
        print(f"  fold held_out=seed{held_out}: train={len(fold_train)} rows, "
              f"oof-predict={len(fold_test)} rows")

    # Final refit on all train_seeds -> genuinely out-of-fold predictions for test_seed
    train_all = labeled[labeled["seed"].isin(train_seeds)]
    test_rows = labeled[labeled["seed"] == test_seed]
    overlap = set(train_all["episode_idx"]) & set(test_rows["episode_idx"])
    assert len(overlap) == 0, f"LEAKAGE: train/test episode_idx overlap = {len(overlap)}"

    final_model = RandomForestClassifier(**rf_kwargs)
    final_model.fit(train_all[feature_cols].values, train_all[label_col].values)
    proba_test = final_model.predict_proba(test_rows[feature_cols].values)
    proba_test_df = pd.DataFrame(proba_test, columns=final_model.classes_, index=test_rows.index)
    proba_test_df = proba_test_df.reindex(columns=classes_sorted, fill_value=0.0)
    fold_frames.append(_assemble(test_rows, proba_test_df, prefix))
    proba_frames.append(_with_key(test_rows, proba_test_df))
    print(f"  final refit (all train_seeds) -> predict seed{test_seed}: "
          f"train={len(train_all)} rows, predict={len(test_rows)} rows\n")

    preds = pd.concat(fold_frames, ignore_index=True)
    proba = pd.concat(proba_frames, ignore_index=True)
    assert len(preds) == len(labeled), \
        f"Row count mismatch: {len(preds)} predictions vs {len(labeled)} labeled input rows"
    return preds, final_model, proba


def _assemble(sub_df, proba_df, prefix):
    pred_class = proba_df.idxmax(axis=1)
    confidence = proba_df.max(axis=1)
    frame = sub_df[KEY_COLS].copy()
    frame[f"{prefix}_pred_class"] = pred_class.values
    frame[f"{prefix}_confidence"] = confidence.values
    return frame


def _with_key(sub_df, proba_df):
    frame = sub_df[KEY_COLS].copy()
    proba_df = proba_df.copy()
    proba_df.index = frame.index
    return pd.concat([frame, proba_df], axis=1)


def onehot_upstream(preds, prefix, all_classes):
    """
    One-hot encode `{prefix}_pred_class` (fixed column set = `all_classes`,
    so unseen-in-this-split classes still get a zero column) and keep
    `{prefix}_confidence` alongside -- the exact "predicted class (one-hot)
    + one confidence scalar" schema CLAUDE.md's chaining decision specifies.
    """
    onehot = pd.get_dummies(preds[f"{prefix}_pred_class"], prefix=prefix)
    onehot = onehot.reindex(columns=[f"{prefix}_{c}" for c in all_classes], fill_value=0)
    out = pd.concat([preds[KEY_COLS + [f"{prefix}_confidence"]], onehot], axis=1)
    return out
