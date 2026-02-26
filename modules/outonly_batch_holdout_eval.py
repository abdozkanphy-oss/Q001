# modules/outonly_batch_holdout_eval.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def select_holdout_batches(
    seg: Sequence[Any],
    *,
    holdout_k: int = 3,
    min_points_per_batch: int = 50,
    exclude_values: Optional[Sequence[str]] = ("MISSING",),
) -> List[str]:
    """
    Select last-K batches by last appearance index in the supervised sample stream.

    Returns a list of segment ids (as strings).
    """
    if seg is None:
        return []
    seg = np.asarray(seg, dtype=object)
    exclude = set(str(x) for x in (exclude_values or []))

    seg_vals, seg_counts = np.unique(seg, return_counts=True)
    ok = [
        str(s)
        for (s, c) in zip(seg_vals, seg_counts)
        if int(c) >= int(min_points_per_batch) and str(s) not in exclude
    ]
    if len(ok) < (int(holdout_k) + 1):
        return []

    last_idx = {}
    for i, s in enumerate(seg):
        ss = str(s)
        if ss in ok:
            last_idx[ss] = i

    ok_sorted = sorted(ok, key=lambda s: int(last_idx.get(s, -1)))
    return ok_sorted[-int(holdout_k):]


def batch_holdout_eval(
    model,
    X: np.ndarray,
    y: np.ndarray,
    baseline_lag0: np.ndarray,
    seg: Sequence[Any],
    *,
    holdout_k: int = 3,
    min_points_per_batch: int = 50,
    min_test: int = 20,
) -> Dict[str, Any]:
    """
    Evaluate a fitted model against lag0 persistence baseline on last-K holdout batches.

    - `model` must implement predict(X)
    - `baseline_lag0` is y(t) aligned to y(t+h) targets
    """
    seg = np.asarray(seg, dtype=object)
    test_segs = select_holdout_batches(seg, holdout_k=holdout_k, min_points_per_batch=min_points_per_batch)
    if not test_segs:
        return {"ok": False, "reason": "insufficient_batches"}

    is_test = np.array([str(s) in set(test_segs) for s in seg], dtype=bool)
    if int(is_test.sum()) < int(min_test) or int((~is_test).sum()) < 1:
        return {"ok": False, "reason": "too_few_test_points", "n_test": int(is_test.sum())}

    pred = model.predict(X[is_test])

    y_te = y[is_test]
    b0 = baseline_lag0[is_test]

    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(mean_squared_error(y_te, pred, squared=False))

    b_mae = float(mean_absolute_error(y_te, b0))
    b_rmse = float(mean_squared_error(y_te, b0, squared=False))

    return {
        "ok": True,
        "mode": "batch_holdout",
        "holdout_k": int(holdout_k),
        "test_segs": list(test_segs),
        "n_test": int(len(y_te)),
        "model_mae": mae,
        "model_rmse": rmse,
        "baseline_mae_lag0": b_mae,
        "baseline_rmse_lag0": b_rmse,
        "lift_vs_lag0": float(b_mae - mae),
    }
