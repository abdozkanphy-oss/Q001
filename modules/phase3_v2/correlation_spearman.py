"""modules/phase3_v2/correlation_spearman.py

Spearman correlation computation for Phase3V2 window finalization.

Output format matches Cassandra writers used in the project:
list[ {row_sensor: {col_sensor: corr_value}} ]

Notes:
- Missing values are treated as NaN.
- Constant columns and insufficient overlap yield NaN correlations which are
  converted to 0.0 (keeps matrix dense and stable for visualization).
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def _safe_float(x):
    try:
        if x is None:
            return None
        xf = float(x)
        if xf != xf:  # NaN
            return None
        return xf
    except Exception:
        return None


def compute_spearman_correlation_data(
    frames,
    *,
    min_overlap: int = 5,
    round_ndigits: int = 6,
) -> Tuple[List[Dict[str, Dict[str, float]]], List[str]]:
    """Compute Spearman correlation matrix from frames.

    Returns (correlation_data, sensors_order).

    correlation_data: list of dict rows {row_sensor: {col_sensor: corr}}
    sensors_order: stable order used.
    """
    # Lazy import pandas to keep import costs out of hot path.
    import pandas as pd

    sensors = set()
    rows = []
    for fr in frames or []:
        sv = getattr(fr, "sensor_values", None) or {}
        row = {}
        for k, v in sv.items():
            sensors.add(str(k))
            fv = _safe_float(v)
            if fv is not None:
                row[str(k)] = fv
        rows.append(row)

    sensors_order = sorted(sensors)
    if not sensors_order:
        return [], []

    df = pd.DataFrame(rows, columns=sensors_order)

    # pairwise spearman; enforce minimal overlap
    try:
        corr = df.corr(method="spearman", min_periods=int(min_overlap))
    except TypeError:
        # older pandas without min_periods for spearman
        corr = df.corr(method="spearman")

    # Ensure stable ordering and fill NaN
    corr = corr.reindex(index=sensors_order, columns=sensors_order)
    corr = corr.fillna(0.0)

    # Diagonal should always be 1.0 for a correlation matrix visualization
    for s in sensors_order:
        try:
            corr.loc[s, s] = 1.0
        except Exception:
            pass

    # Build Cassandra-friendly structure
    out: List[Dict[str, Dict[str, float]]] = []
    for r in sensors_order:
        row_map: Dict[str, float] = {}
        for c in sensors_order:
            try:
                v = float(corr.loc[r, c])
            except Exception:
                v = 0.0
            if round_ndigits is not None:
                try:
                    v = round(v, int(round_ndigits))
                except Exception:
                    pass
            row_map[c] = v
        out.append({r: row_map})

    return out, sensors_order
