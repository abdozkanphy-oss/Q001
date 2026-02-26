"""modules/phase3_v2/feature_importance_engine.py

Feature-importance-on-finalize executor for Phase3V2.

This computes feature importances when a window is finalized (task/batch/ws) and
writes them to Cassandra.

Implementation is intentionally conservative:
- FI runs only on finalize events (never per message).
- Requires a trained MIMO RF model artifact for the ws+stock context.
- Uses permutation importance over a bounded number of top-ranked features.
- Aggregates lagged feature importances back to the base sensor name.

Persistence:
- Writes to legacy scada_feature_importance_values by default.
- WS window is stored under a synthetic prod_order_reference_no =
  "WS|<ws_uid>|<stock_key>" to avoid key collisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.phase3_v2.mimo_rf_registry import get_mimo_rf_registry, load_joblib_model


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        xf = float(x)
        if xf != xf:  # NaN
            return None
        return xf
    except Exception:
        return None


def _ts_from_ms(ms: int):
    import datetime

    try:
        return datetime.datetime.utcfromtimestamp(float(ms) / 1000.0)
    except Exception:
        return datetime.datetime.utcnow()


def _build_message_from_event(ev) -> Dict[str, Any]:
    fr_last = (ev.frames[-1] if getattr(ev, "frames", None) else None)
    meta = getattr(fr_last, "message_meta", None) or {}

    prod_no = meta.get("produced_stock_no")
    prod_nm = meta.get("produced_stock_name")

    msg: Dict[str, Any] = {
        "customer": meta.get("customer"),
        "plId": meta.get("plId"),
        "wcNo": meta.get("wcNo"),
        "wcNm": meta.get("wcNm"),
        "wsNo": meta.get("wsNo"),
        "wsNm": meta.get("wsNm"),
        "opNo": meta.get("opNo"),
        "opNm": meta.get("opNm"),
        "joRef": meta.get("joRef"),
        "job_order_reference_no": meta.get("joRef"),
        "refNo": str(ev.batch_root or "0"),
        "prod_order_reference_no": str(ev.batch_root or "0"),
        "joOpId": str(ev.phase_id or "0"),
        "proces_no": str(ev.phase_id or "0"),
        "crDt": int(ev.end_ts_ms),
        "measDt": int(ev.end_ts_ms),
        "produced_stock_no": prod_no,
        "produced_stock_name": prod_nm,
        "output_stock_no": prod_no,
        "output_stock_name": prod_nm,
    }

    if prod_no or prod_nm:
        msg["prodList"] = [{"stNo": prod_no, "stNm": prod_nm}]

    return msg


def _parse_base_and_lag(feature_name: str) -> Tuple[str, int]:
    s = str(feature_name)
    # expected patterns: "<sensor>__lag<k>" or "<sensor>__lag_<k>"
    if "__lag" in s:
        base, rest = s.split("__lag", 1)
        rest = rest.lstrip("_")
        try:
            return base, int(rest)
        except Exception:
            return base, 0
    return s, 0


def _extract_static_feature_importances(model, n_features: int) -> Optional[np.ndarray]:
    """Attempt to extract RF-style feature_importances_."""
    try:
        if hasattr(model, "feature_importances_"):
            arr = np.asarray(getattr(model, "feature_importances_"), dtype=float)
            return arr if arr.shape[0] == n_features else None
    except Exception:
        pass

    # MultiOutputRegressor: estimators_ list
    try:
        ests = getattr(model, "estimators_", None)
        if ests:
            vals = []
            for e in ests:
                if hasattr(e, "feature_importances_"):
                    fi = np.asarray(getattr(e, "feature_importances_"), dtype=float)
                    if fi.shape[0] == n_features:
                        vals.append(fi)
            if vals:
                return np.mean(np.stack(vals, axis=0), axis=0)
    except Exception:
        pass

    # pipeline-like
    try:
        steps = getattr(model, "named_steps", None)
        if steps:
            # try last step first
            for _, step in list(steps.items())[::-1]:
                arr = _extract_static_feature_importances(step, n_features)
                if arr is not None:
                    return arr
    except Exception:
        pass

    return None


def _build_xy(frames, art) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build lagged X, Y from frames using artifact feature_names and targets."""
    frames_sorted = sorted(list(frames or []), key=lambda f: int(getattr(f, "ts_ms", 0)))

    mapping = [_parse_base_and_lag(fn) for fn in art.feature_names]
    base_by_feat = [b for (b, _) in mapping]

    n_lags = int(getattr(art, "n_lags", 0) or 0)
    targets = list(getattr(art, "targets", []) or [])

    X_rows = []
    Y_rows = []

    for i in range(n_lags, len(frames_sorted)):
        x = []
        for (base, lag) in mapping:
            j = i - int(lag)
            if j < 0:
                x.append(0.0)
                continue
            v = _safe_float((frames_sorted[j].sensor_values or {}).get(base))
            x.append(float(v) if v is not None else 0.0)

        y = []
        cur = frames_sorted[i].sensor_values or {}
        for t in targets:
            vv = _safe_float(cur.get(t))
            y.append(float(vv) if vv is not None else 0.0)

        X_rows.append(x)
        Y_rows.append(y)

    if not X_rows or not targets:
        return np.zeros((0, 0)), np.zeros((0, 0)), base_by_feat, targets

    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_rows, dtype=float)
    return X, Y, base_by_feat, targets


def _perm_importance(model, X: np.ndarray, Y: np.ndarray, *, top_k: int, seed: int = 13) -> np.ndarray:
    """Permutation importance on MAE aggregated over outputs."""
    n_samples, n_features = X.shape
    if n_samples <= 0 or n_features <= 0:
        return np.zeros((n_features,), dtype=float)

    # baseline
    pred = model.predict(X)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    mae0 = float(np.mean(np.abs(pred - Y)))

    fi_static = _extract_static_feature_importances(model, n_features)
    if fi_static is None:
        cand = list(range(n_features))
    else:
        cand = list(np.argsort(-fi_static))

    k = int(max(1, min(int(top_k), len(cand))))
    cand = cand[:k]

    rng = np.random.default_rng(int(seed))
    out = np.zeros((n_features,), dtype=float)

    # compute only for candidate features
    for j in cand:
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        predp = model.predict(Xp)
        predp = np.asarray(predp, dtype=float)
        if predp.ndim == 1:
            predp = predp.reshape(-1, 1)
        maep = float(np.mean(np.abs(predp - Y)))
        out[j] = max(0.0, maep - mae0)

    return out


def _aggregate_to_base(fi_by_feature: np.ndarray, base_by_feat: List[str]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for v, base in zip(fi_by_feature.tolist(), base_by_feat):
        b = str(base)
        agg[b] = agg.get(b, 0.0) + float(v)
    return agg


def _normalize_map(m: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(0.0, float(v)) for v in m.values()))
    if s <= 0.0:
        # uniform
        keys = list(m.keys())
        if not keys:
            return {}
        u = 1.0 / float(len(keys))
        return {k: u for k in keys}
    return {k: float(max(0.0, float(v)) / s) for k, v in m.items()}


@dataclass
class FIPersistResult:
    computed: bool
    wrote_legacy: bool
    reason: str


@dataclass
class _TaskFIAccum:
    sum_in: Dict[str, float]
    sum_out: Dict[str, float]
    n_windows: int


class Phase3V2FeatureImportanceEngine:
    def __init__(self, cfg, *, models_dir: str):
        self.cfg = cfg
        self.models_dir = str(models_dir)

        self.enabled = bool(getattr(cfg, "phase3v2_fi_enabled", True))
        self.persist_enabled = bool(getattr(cfg, "phase3_derived_persist_enabled", True))

        self.min_frames = int(getattr(cfg, "phase3v2_fi_min_frames", 80) or 80)
        self.max_features = int(getattr(cfg, "phase3v2_fi_max_features", 50) or 50)
        self.seed = int(getattr(cfg, "phase3v2_fi_seed", 13) or 13)

        self.algorithm = str(getattr(cfg, "phase3v2_fi_algorithm", "PERM_IMPORTANCE_MIMO_RF") or "PERM_IMPORTANCE_MIMO_RF")
        self.write_legacy = bool(getattr(cfg, "phase3v2_fi_write_legacy", True))

        # Temporary: write a "task-window average" row at batch finalization.
        self.write_taskavg_on_batch_finalize = bool(
            getattr(cfg, "phase3v2_write_taskavg_on_batch_finalize", True)
        )
        self.taskavg_bump_ms = int(getattr(cfg, "phase3v2_taskavg_bump_ms", 1) or 1)
        self.taskavg_bump_ms = int(max(1, min(1000, self.taskavg_bump_ms)))

        self._task_accum: Dict[Tuple[str, str, str], _TaskFIAccum] = {}

        self._cache: Dict[Tuple[str, str, int], Tuple[Any, Any]] = {}

    def _accum_key(self, ev) -> Tuple[str, str, str]:
        return (str(ev.ws_uid), str(ev.stock_key), str(ev.batch_root or "0"))

    def _accum_add(self, ev, imp_in: Dict[str, float], imp_out: Dict[str, float]) -> None:
        if not imp_in and not imp_out:
            return
        k = self._accum_key(ev)
        acc = self._task_accum.get(k)
        if acc is None:
            acc = _TaskFIAccum(sum_in={}, sum_out={}, n_windows=0)
            self._task_accum[k] = acc

        for kk, vv in (imp_in or {}).items():
            try:
                acc.sum_in[str(kk)] = acc.sum_in.get(str(kk), 0.0) + float(vv)
            except Exception:
                continue
        for kk, vv in (imp_out or {}).items():
            try:
                acc.sum_out[str(kk)] = acc.sum_out.get(str(kk), 0.0) + float(vv)
            except Exception:
                continue
        acc.n_windows += 1

    def _accum_pop_taskavg(self, ev) -> Optional[_TaskFIAccum]:
        return self._task_accum.pop(self._accum_key(ev), None)

    def _load_model(self, ev) -> Tuple[Optional[Any], Optional[Any], str]:
        fr_last = (ev.frames[-1] if getattr(ev, "frames", None) else None)
        meta = getattr(fr_last, "message_meta", None) or {}

        rsec = int(meta.get("resample_sec") or 0)
        key = (str(ev.ws_uid), str(ev.stock_key), int(rsec))
        if key in self._cache:
            model, art = self._cache[key]
            return model, art, "cache"

        registry = get_mimo_rf_registry(self.models_dir)

        stock_candidates = []
        for c in [meta.get("produced_stock_no"), meta.get("produced_stock_name"), ev.stock_key]:
            if c is None:
                continue
            s = str(c).strip()
            if s and s not in stock_candidates:
                stock_candidates.append(s)

        op_candidates = []
        for c in [meta.get("opTc")]:
            if c is None:
                continue
            s = str(c).strip()
            if s and s not in op_candidates:
                op_candidates.append(s)

        art = registry.find_best(
            wsuid=str(ev.ws_uid),
            stock_candidates=stock_candidates,
            op_candidates=op_candidates,
            resample_sec=int(rsec) if int(rsec) > 0 else None,
        )
        if art is None:
            return None, None, "no_model"

        model = load_joblib_model(art.model_path)
        self._cache[key] = (model, art)
        return model, art, "loaded"

    def on_window_event(self, ev, *, log=None) -> FIPersistResult:
        if not self.enabled:
            return FIPersistResult(False, False, "disabled")
        frames = list(getattr(ev, "frames", ()) or ())
        # defensive ordering
        try:
            frames = sorted(frames, key=lambda f: int(getattr(f, "ts_ms", 0) or 0))
        except Exception:
            pass

        model, art, why = self._load_model(ev)
        if model is None or art is None:
            return FIPersistResult(False, False, why)

        # Require enough frames to build a meaningful (X,Y) given n_lags.
        try:
            min_xy_rows = int(getattr(self.cfg, "phase3v2_fi_min_xy_rows", 10) or 10)
        except Exception:
            min_xy_rows = 10
        n_lags = int(getattr(art, "n_lags", 0) or 0)
        needed = max(int(self.min_frames), int(n_lags) + int(max(1, min_xy_rows)))
        if len(frames) < needed:
            return FIPersistResult(False, False, f"insufficient_frames<{needed}")

        X, Y, base_by_feat, targets = _build_xy(frames, art)
        if X.size == 0 or Y.size == 0:
            return FIPersistResult(False, False, "empty_xy")

        # permutation importance
        fi_feat = _perm_importance(model, X, Y, top_k=self.max_features, seed=self.seed)
        agg = _aggregate_to_base(fi_feat, base_by_feat)
        agg = _normalize_map(agg)

        # output importance: normalized variance (fallback uniform)
        try:
            var = np.var(Y, axis=0)
            var_map = {str(t): float(v) for t, v in zip(targets, var.tolist())}
            var_map = _normalize_map(var_map)
        except Exception:
            var_map = {str(t): 1.0 / float(len(targets)) for t in targets}

        imp_input = {f"in_{k}": float(v) for k, v in sorted(agg.items())}
        imp_output = {f"out_{k}": float(v) for k, v in sorted(var_map.items())}

        # accumulate task-window FI for batch-tail summary
        if ev.kind == "FINALIZE_TASK" and self.write_taskavg_on_batch_finalize:
            try:
                self._accum_add(ev, imp_input, imp_output)
            except Exception as e:
                if log:
                    log.error(f"[phase3v2][fi] task accum add failed: {e}", exc_info=True)

        msg = _build_message_from_event(ev)
        if ev.kind == "FINALIZE_TASK":
            msg["joOpId"] = str(ev.phase_id or "0")
            msg["proces_no"] = msg["joOpId"]
        elif ev.kind == "FINALIZE_BATCH":
            msg["joOpId"] = f"BATCH|{ev.batch_root or '0'}"
            msg["proces_no"] = msg["joOpId"]
        elif ev.kind == "FINALIZE_WS":
            msg["joOpId"] = f"WS|{ev.ws_uid}"
            msg["proces_no"] = msg["joOpId"]
            # ensure partition separation per ws+stock
            msg["prod_order_reference_no"] = f"WS|{ev.ws_uid}|{ev.stock_key}"
            msg["refNo"] = msg["prod_order_reference_no"]

        wrote_legacy = False
        try:
            if self.persist_enabled and self.write_legacy:
                from cassandra_utils.models.scada_feature_importance_values import ScadaFeatureImportanceValues

                ScadaFeatureImportanceValues.saveData(
                    msg,
                    imp_input,
                    imp_output,
                    algorithm=self.algorithm,
                    now_ts=_ts_from_ms(int(ev.end_ts_ms)),
                )
                wrote_legacy = True
        except Exception as e:
            if log:
                log.error(f"[phase3v2][fi] legacy write failed: {e}", exc_info=True)

        # Temporary: on batch finalize, write an extra "task average" row with bumped partition_date.
        if (
            ev.kind == "FINALIZE_BATCH"
            and self.write_taskavg_on_batch_finalize
            and self.persist_enabled
            and self.write_legacy
        ):
            try:
                acc = self._accum_pop_taskavg(ev)
                if acc is not None and int(acc.n_windows) > 0:
                    nwin = float(max(1, int(acc.n_windows)))
                    avg_in = {k: float(v) / nwin for k, v in (acc.sum_in or {}).items()}
                    avg_out = {k: float(v) / nwin for k, v in (acc.sum_out or {}).items()}
                    # renormalize for stable visualization
                    avg_in = _normalize_map(avg_in)
                    avg_out = _normalize_map(avg_out)

                    msg_avg = _build_message_from_event(ev)
                    msg_avg["crDt"] = int(ev.end_ts_ms) + int(self.taskavg_bump_ms)
                    msg_avg["measDt"] = msg_avg["crDt"]
                    msg_avg["joOpId"] = f"BATCHAVG|{ev.batch_root or '0'}"
                    msg_avg["proces_no"] = msg_avg["joOpId"]

                    from cassandra_utils.models.scada_feature_importance_values import ScadaFeatureImportanceValues

                    ScadaFeatureImportanceValues.saveData(
                        msg_avg,
                        avg_in,
                        avg_out,
                        algorithm=f"{self.algorithm}_TASKAVG",
                        now_ts=_ts_from_ms(int(msg_avg["crDt"])),
                        p3_1_log=log,
                    )
            except Exception as e:
                if log:
                    log.error(f"[phase3v2][fi] batch taskavg write failed: {e}", exc_info=True)

        reason = "ok"
        if not self.persist_enabled:
            if ev.kind == "FINALIZE_BATCH" and self.write_taskavg_on_batch_finalize:
                try:
                    _ = self._accum_pop_taskavg(ev)
                except Exception:
                    pass
            reason = "persist_disabled"
        elif self.write_legacy and not wrote_legacy:
            reason = "write_failed"
        elif not self.write_legacy:
            reason = "write_disabled"

        return FIPersistResult(True, wrote_legacy, reason)
