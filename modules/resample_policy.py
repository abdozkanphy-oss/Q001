# modules/resample_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResamplePolicyResult:
    """Workstation-level resample policy recommendation.

    This is intentionally lightweight and explainable. It is meant to be used by both
    Phase2 and Phase3 (training + inference routing), and must be derived from event-time
    measurements (measDt / measurement_date), not wall-clock.
    """

    recommended_resample_sec: int
    candidate_resample_sec: List[int]
    max_gap_sec: int
    allowed_lateness_sec: int
    min_frames_for_infer: int
    min_frames_for_train: int
    flags: List[str]
    debug: Dict[str, Any]


def _to_epoch_seconds(ts: pd.Series) -> np.ndarray:
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    s = s.dropna()
    if s.empty:
        return np.array([], dtype=np.int64)
    # seconds since epoch
    return (s.astype("int64") // 1_000_000_000).astype(np.int64).to_numpy()


def _run_lengths(mask_sorted_unique: np.ndarray) -> List[int]:
    """Compute run lengths of consecutive integers from sorted unique integer array."""
    if mask_sorted_unique.size == 0:
        return []
    diffs = np.diff(mask_sorted_unique)
    breaks = np.where(diffs != 1)[0]
    # indices of run starts
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, mask_sorted_unique.size - 1]
    return (ends - starts + 1).astype(int).tolist()


def recommend_resample_policy(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    sensor_col: str = "sensor",
    val_col: str = "val",
    candidates: Optional[Iterable[int]] = None,
    max_sensors: int = 50,
    seq_len: int = 20,
    min_total_points: int = 200,
) -> ResamplePolicyResult:
    """Recommend a workstation-level resample policy.

    Inputs:
      df: long-format dataframe with at least (ts_col, sensor_col, val_col)

    Output:
      ResamplePolicyResult with a recommended resample_sec and supporting debug metrics.

    Notes:
      - We do not trust nominal cadences; we infer from event-time deltas.
      - We avoid a continuous optimization and instead score a small discrete candidate set.
      - This is v1: deterministic, bounded, explainable. We can evolve scoring later.
    """

    if candidates is None:
        candidates = [5, 10, 15, 30, 60, 120, 300]
    cand = [int(x) for x in candidates if int(x) > 0]
    cand = sorted(set(cand))
    if not cand:
        cand = [60]

    # Defensive defaults
    default_mid = int(cand[len(cand) // 2])
    default = ResamplePolicyResult(
        recommended_resample_sec=default_mid,
        candidate_resample_sec=cand,
        max_gap_sec=int(6 * default_mid),
        allowed_lateness_sec=int(2 * default_mid),
        min_frames_for_infer=max(40, 2 * seq_len),
        min_frames_for_train=max(200, 10 * seq_len),
        flags=["DEFAULT_NO_DATA"],
        debug={"reason": "no_or_insufficient_data"},
    )

    if df is None or df.empty or ts_col not in df.columns:
        return default

    # Bound sensors to keep runtime deterministic.
    cols = [ts_col]
    if sensor_col in df.columns:
        cols.append(sensor_col)
    if val_col in df.columns:
        cols.append(val_col)
    sdf = df[cols].copy()
    sdf[ts_col] = pd.to_datetime(sdf[ts_col], errors="coerce", utc=True)
    sdf = sdf.dropna(subset=[ts_col])
    if sdf.empty:
        return default

    if sensor_col in sdf.columns:
        top_sensors = (
            sdf[sensor_col]
            .fillna("")
            .astype(str)
            .value_counts()
            .head(int(max_sensors))
            .index
            .tolist()
        )
        sdf = sdf[sdf[sensor_col].astype(str).isin(top_sensors)]

    low_data = bool(len(sdf) < int(min_total_points))

    ts_sec = _to_epoch_seconds(sdf[ts_col])
    if ts_sec.size < 2:
        return default

    start_sec = int(ts_sec.min())
    end_sec = int(ts_sec.max())
    span_sec = max(1, end_sec - start_sec)

    # Infer effective cadence from per-sensor median dt (robust).
    per_sensor_dt_med: List[Tuple[float, int]] = []  # (median_dt, count)
    dup_ratio_by_sensor: List[Tuple[float, int]] = []  # (dup_ratio, count)

    if sensor_col in sdf.columns and val_col in sdf.columns:
        for _, g in sdf.groupby(sensor_col, sort=False):
            g = g.sort_values(ts_col)
            tsec = _to_epoch_seconds(g[ts_col])
            if tsec.size < 3:
                continue
            d = np.diff(tsec)
            d = d[d > 0]
            if d.size:
                per_sensor_dt_med.append((float(np.median(d)), int(tsec.size)))

            # Detect likely upsampled feeds: mostly identical values across consecutive points.
            try:
                v = pd.to_numeric(g[val_col], errors="coerce").to_numpy()
                if v.size >= 3:
                    same = np.isfinite(v[1:]) & np.isfinite(v[:-1]) & (v[1:] == v[:-1])
                    dup_ratio = float(same.mean()) if same.size else 0.0
                    dup_ratio_by_sensor.append((dup_ratio, int(v.size)))
            except Exception:
                pass

    if per_sensor_dt_med:
        # Weighted median by sensor count
        per_sensor_dt_med.sort(key=lambda x: x[0])
        total_w = sum(w for _, w in per_sensor_dt_med)
        acc = 0
        eff_dt = per_sensor_dt_med[0][0]
        for dt, w in per_sensor_dt_med:
            acc += w
            if acc >= total_w / 2:
                eff_dt = dt
                break
        eff_dt = max(1.0, float(eff_dt))
    else:
        # Fallback: global median dt
        d = np.diff(np.sort(ts_sec))
        d = d[d > 0]
        eff_dt = float(np.median(d)) if d.size else 60.0
        eff_dt = max(1.0, eff_dt)

    upsample_flag = False
    if dup_ratio_by_sensor:
        num = sum(r * w for r, w in dup_ratio_by_sensor)
        den = sum(w for _, w in dup_ratio_by_sensor)
        avg_dup_ratio = float(num / max(1, den))
        # Heuristic: mostly duplicates and effective dt is slow -> likely upsampled representation.
        if avg_dup_ratio >= 0.9 and eff_dt >= 120:
            upsample_flag = True
    else:
        avg_dup_ratio = None

    # Score candidates.
    scores: Dict[int, Dict[str, Any]] = {}
    flags: List[str] = []
    if low_data:
        flags.append("LOW_DATA")
    if upsample_flag:
        flags.append("UPSAMPLED_FEED_SUSPECTED")

    ts_sorted = np.sort(ts_sec)
    for r in cand:
        total_bins = int(span_sec // r) + 1
        bins_all = np.unique(((ts_sorted - start_sec) // r).astype(np.int64))
        union_coverage = float(len(bins_all) / max(1, total_bins))

        run_lens = _run_lengths(bins_all)
        seq_yield = int(sum(max(0, rl - seq_len + 1) for rl in run_lens))
        max_run = int(max(run_lens) if run_lens else 0)

        oversample_pen = 0.0
        if r < eff_dt:
            oversample_pen = float((eff_dt / max(1.0, r)) - 1.0)
        if upsample_flag and r < 120:
            oversample_pen *= 2.0

        duty_pen = float(max(0.0, 0.5 - union_coverage))

        score = (
            2.0 * union_coverage
            + 0.0005 * float(seq_yield)
            - 0.25 * oversample_pen
            - 0.25 * duty_pen
        )

        scores[int(r)] = {
            "score": float(score),
            "union_coverage": union_coverage,
            "seq_yield": int(seq_yield),
            "max_run_bins": max_run,
            "total_bins": int(total_bins),
            "n_points": int(ts_sorted.size),
            "oversample_pen": float(oversample_pen),
            "duty_pen": float(duty_pen),
        }

    best_r = max(scores.items(), key=lambda kv: (kv[1]["score"], -kv[0]))[0]

    recommended = int(best_r)
    max_gap_sec = int(min(max(6 * recommended, 30), 900))
    allowed_lateness_sec = int(min(max(2 * recommended, 10), 300))

    min_frames_for_infer = int(max(2 * seq_len, 40))
    min_frames_for_train = int(max(10 * seq_len, 200))

    dbg = {
        "effective_dt_sec": float(eff_dt),
        "avg_dup_ratio": None if avg_dup_ratio is None else float(avg_dup_ratio),
        "span_sec": int(span_sec),
        "candidates": cand,
        "scores": scores,
    }

    return ResamplePolicyResult(
        recommended_resample_sec=recommended,
        candidate_resample_sec=cand,
        max_gap_sec=max_gap_sec,
        allowed_lateness_sec=allowed_lateness_sec,
        min_frames_for_infer=min_frames_for_infer,
        min_frames_for_train=min_frames_for_train,
        flags=flags,
        debug=dbg,
    )


def policy_to_dict(p: ResamplePolicyResult) -> Dict[str, Any]:
    return {
        "recommended_resample_sec": int(p.recommended_resample_sec),
        "candidate_resample_sec": [int(x) for x in (p.candidate_resample_sec or [])],
        "max_gap_sec": int(p.max_gap_sec),
        "allowed_lateness_sec": int(p.allowed_lateness_sec),
        "min_frames_for_infer": int(p.min_frames_for_infer),
        "min_frames_for_train": int(p.min_frames_for_train),
        "flags": list(p.flags or []),
        "debug": dict(p.debug or {}),
    }
