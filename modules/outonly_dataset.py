from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class OutOnlyDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    index_ts_ms: np.ndarray  # timestamp of row (bucket dt)


def _parse_feature_name(name: str) -> Tuple[str, int]:
    """Return (sensor_name, lag_k). For UNI mode, sensor_name may be ''."""
    s = str(name)
    lag = None
    # common patterns:
    # SENSOR__lag_3, SENSOR__lag3, SENSOR__lag_03
    # lag_3, lag3
    if "__" in s:
        left, right = s.split("__", 1)
        base = left
        tail = right
    else:
        base = ""
        tail = s
    tail = tail.lower().replace("-", "_")
    # find 'lag' then digits
    if "lag" in tail:
        idx = tail.find("lag")
        digits = "".join([c for c in tail[idx + 3 :] if c.isdigit()])
        if digits:
            lag = int(digits)
    if lag is None:
        # fallback: treat as lag0
        lag = 0
    return base, lag


def build_outonly_dataset_from_wide(
    wide: pd.DataFrame,
    *,
    target_col: str,
    horizon_steps: int,
    feature_names: Optional[List[str]] = None,
) -> OutOnlyDataset:
    """Build (X,y) for OUT_ONLY evaluation from a wide bucketed dataframe.

    - wide index: epoch ms (int) or datetime
    - columns: sensor names
    """
    df = wide.copy()
    # Ensure monotonically increasing index
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # target y(t+h)
    h = int(max(1, horizon_steps))
    y = df[target_col].shift(-h)

    # determine features
    feats = list(feature_names or [])
    if not feats:
        # default: lag0..lag5 of target
        feats = [f"lag{k}" for k in range(6)]

    X_cols = []
    X = np.zeros((len(df), len(feats)), dtype="float64")
    for j, fn in enumerate(feats):
        sensor, lag = _parse_feature_name(fn)
        col = target_col if sensor == "" else sensor
        if col not in df.columns:
            X[:, j] = np.nan
        else:
            X[:, j] = df[col].shift(int(lag)).to_numpy(dtype="float64")
        X_cols.append(fn)

    # align and drop rows with NaN in y
    y_arr = y.to_numpy(dtype="float64")
    ts = df.index.to_numpy()
    m = np.isfinite(y_arr)
    X = X[m]
    y_arr = y_arr[m]
    ts = ts[m]
    return OutOnlyDataset(X=X, y=y_arr, feature_names=X_cols, index_ts_ms=ts)
