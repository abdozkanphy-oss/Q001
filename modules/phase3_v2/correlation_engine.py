"""modules/phase3_v2/correlation_engine.py

Correlation-on-finalize executor for Phase3V2.

This runs correlation computations *only* on window finalize events emitted by
WindowManager and writes results to Cassandra.

Persistence policy:
- TASK and BATCH windows: write to both the legacy table (scada_correlation_matrix)
  and the v2 table (scada_correlation_matrix_ws_stock_batch) by default.
- WS (workstation partition) window: write to v2 global table
  (scada_correlation_matrix_ws_stock_global). Legacy table cannot partition by
  ws+stock without synthetic keys.

All writes are gated by phase3_derived_persist_enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from modules.phase3_v2.correlation_spearman import compute_spearman_correlation_data


def _ts_from_ms(ms: int):
    import datetime

    try:
        return datetime.datetime.utcfromtimestamp(float(ms) / 1000.0)
    except Exception:
        return datetime.datetime.utcnow()


def _build_message_from_event(ev) -> Dict[str, Any]:
    """Build a minimal message dict compatible with existing Cassandra writers."""
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

    # legacy writers prefer prodList[*].stNo/stNm
    if prod_no or prod_nm:
        msg["prodList"] = [{"stNo": prod_no, "stNm": prod_nm}]

    return msg


@dataclass
class CorrPersistResult:
    computed: bool
    wrote_legacy: bool
    wrote_v2_batch: bool
    wrote_v2_global: bool
    reason: str


@dataclass
class _TaskCorrAccum:
    sensors: list
    sum_mat: np.ndarray
    n_windows: int


class Phase3V2CorrelationEngine:
    def __init__(self, cfg, *, models_dir: Optional[str] = None):
        self.cfg = cfg

        self.enabled = bool(getattr(cfg, "phase3v2_corr_enabled", True))
        self.persist_enabled = bool(getattr(cfg, "phase3_derived_persist_enabled", True))

        self.min_frames = int(getattr(cfg, "phase3v2_corr_min_frames", 30) or 30)
        self.min_overlap = int(getattr(cfg, "phase3v2_corr_min_overlap", 5) or 5)
        self.round_ndigits = int(getattr(cfg, "phase3v2_corr_round_ndigits", 6) or 6)

        self.write_legacy = bool(getattr(cfg, "phase3v2_corr_write_legacy", True))
        self.write_v2 = bool(getattr(cfg, "phase3v2_corr_write_v2", True))

        # Temporary: write a "task-window average" row at batch finalization.
        # This helps backends that read only the last row. Can be disabled later.
        self.write_taskavg_on_batch_finalize = bool(
            getattr(cfg, "phase3v2_write_taskavg_on_batch_finalize", True)
        )
        self.taskavg_bump_ms = int(getattr(cfg, "phase3v2_taskavg_bump_ms", 1) or 1)
        self.taskavg_bump_ms = int(max(1, min(1000, self.taskavg_bump_ms)))

        # Optional: contrast stretch for the batch tail task-average matrix.
        # By default we rescale off-diagonal values so the largest absolute value maps to 1.0.
        # This is a temporary visualization-oriented workaround and can be disabled later.
        self.taskavg_rescale_enabled = bool(
            getattr(cfg, "phase3v2_taskavg_rescale_enabled", True)
        )
        self.taskavg_rescale_use_offdiag = bool(
            getattr(cfg, "phase3v2_taskavg_rescale_use_offdiag", True)
        )
        self.taskavg_rescale_symmetrize = bool(
            getattr(cfg, "phase3v2_taskavg_rescale_symmetrize", True)
        )
        try:
            self.taskavg_rescale_eps = float(getattr(cfg, "phase3v2_taskavg_rescale_eps", 1e-9) or 1e-9)
        except Exception:
            self.taskavg_rescale_eps = 1e-9

        # Accumulate FINALIZE_TASK correlations per (ws, stock, batch_root)
        self._task_accum: Dict[Tuple[str, str, str], _TaskCorrAccum] = {}

    def _accum_key(self, ev) -> Tuple[str, str, str]:
        return (str(ev.ws_uid), str(ev.stock_key), str(ev.batch_root or "0"))

    def _accum_add(self, ev, corr_data, sensors_order) -> None:
        if not corr_data or not sensors_order:
            return
        k = self._accum_key(ev)
        acc = self._task_accum.get(k)
        if acc is None:
            n = len(sensors_order)
            acc = _TaskCorrAccum(sensors=list(sensors_order), sum_mat=np.zeros((n, n), dtype="float64"), n_windows=0)
            self._task_accum[k] = acc

        # union sensor set (stable sorted)
        union = sorted(set(acc.sensors).union(set(sensors_order)))
        if union != acc.sensors:
            old_idx = {s: i for i, s in enumerate(acc.sensors)}
            new_idx = {s: i for i, s in enumerate(union)}
            new_sum = np.zeros((len(union), len(union)), dtype="float64")
            for si in acc.sensors:
                for sj in acc.sensors:
                    new_sum[new_idx[si], new_idx[sj]] = acc.sum_mat[old_idx[si], old_idx[sj]]
            acc.sensors = union
            acc.sum_mat = new_sum

        idx = {s: i for i, s in enumerate(acc.sensors)}
        # corr_data format: list[{row_sensor: {col_sensor: corr}}]
        for item in corr_data:
            if not isinstance(item, dict):
                continue
            for r, inner in item.items():
                if r not in idx or not isinstance(inner, dict):
                    continue
                ir = idx[r]
                for c, v in inner.items():
                    if c not in idx:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        fv = 0.0
                    acc.sum_mat[ir, idx[c]] += fv

        acc.n_windows += 1

    def _accum_pop_taskavg(self, ev) -> Optional[Tuple[list, np.ndarray, int]]:
        k = self._accum_key(ev)
        acc = self._task_accum.pop(k, None)
        if acc is None or int(acc.n_windows) <= 0:
            return None
        return acc.sensors, acc.sum_mat, int(acc.n_windows)

    def _mat_to_corr_data(self, sensors: list, mat: np.ndarray) -> Any:
        # Ensure diagonal=1 and round to configured digits.
        n = len(sensors)
        out = []
        for i, r in enumerate(sensors):
            row_map = {}
            for j, c in enumerate(sensors):
                if i == j:
                    v = 1.0
                else:
                    try:
                        v = float(mat[i, j])
                    except Exception:
                        v = 0.0
                try:
                    v = round(v, int(self.round_ndigits))
                except Exception:
                    pass
                row_map[str(c)] = float(v)
            out.append({str(r): row_map})
        return out


    def _rescale_taskavg_matrix(self, mat: np.ndarray, *, log=None) -> np.ndarray:
        """Rescale (contrast-stretch) task-average correlation matrix for readability.

        We rescale values into [-1, 1] by dividing by max(|value|).
        By default, the maximum is computed on off-diagonal entries so the diagonal=1
        does not suppress scaling. This is a temporary backend visualization workaround.
        """
        if not self.taskavg_rescale_enabled:
            return mat

        try:
            m = np.array(mat, dtype="float64", copy=True)
        except Exception:
            return mat

        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
            return m

        if self.taskavg_rescale_symmetrize:
            try:
                m = 0.5 * (m + m.T)
            except Exception:
                pass

        n = int(m.shape[0])
        try:
            if self.taskavg_rescale_use_offdiag:
                mask = ~np.eye(n, dtype=bool)
                vals = m[mask]
            else:
                vals = m.reshape(-1)
            max_abs = float(np.nanmax(np.abs(vals))) if vals.size else 0.0
        except Exception:
            max_abs = 0.0

        if not np.isfinite(max_abs) or max_abs < float(self.taskavg_rescale_eps or 1e-9):
            # ensure diagonal is 1 and clip defensively
            try:
                np.fill_diagonal(m, 1.0)
            except Exception:
                pass
            return np.clip(m, -1.0, 1.0)

        try:
            m = m / max_abs
        except Exception:
            return mat

        m = np.clip(m, -1.0, 1.0)
        try:
            np.fill_diagonal(m, 1.0)
        except Exception:
            pass

        if log:
            try:
                log.info(f"[phase3v2][corr] taskavg rescale applied: max_abs={max_abs:.6g}")
            except Exception:
                pass

        return m

    def on_window_event(self, ev, *, log=None) -> CorrPersistResult:
        if not self.enabled:
            return CorrPersistResult(False, False, False, False, "disabled")

        frames = list(getattr(ev, "frames", ()) or ())
        # defensive: always sort by event-time so correlation is reproducible
        try:
            frames = sorted(frames, key=lambda f: int(getattr(f, "ts_ms", 0) or 0))
        except Exception:
            pass
        if len(frames) < self.min_frames:
            return CorrPersistResult(False, False, False, False, f"insufficient_frames<{self.min_frames}")

        corr_data, sensors = compute_spearman_correlation_data(
            frames,
            min_overlap=self.min_overlap,
            round_ndigits=self.round_ndigits,
        )

        if not corr_data:
            return CorrPersistResult(False, False, False, False, "empty")

        msg = _build_message_from_event(ev)

        wrote_legacy = False
        wrote_v2_batch = False
        wrote_v2_global = False

        # Distinguish window type in v2 tables via algorithm name.
        algo_v2 = "SPEARMAN"
        if ev.kind == "FINALIZE_TASK":
            algo_v2 = "SPEARMAN_TASK"
            msg["joOpId"] = str(ev.phase_id or "0")
            msg["proces_no"] = str(ev.phase_id or "0")
        elif ev.kind == "FINALIZE_BATCH":
            algo_v2 = "SPEARMAN_BATCH"
            msg["joOpId"] = f"BATCH|{ev.batch_root or '0'}"
            msg["proces_no"] = msg["joOpId"]
        elif ev.kind == "FINALIZE_WS":
            algo_v2 = "SPEARMAN_WS"
            msg["joOpId"] = f"WS|{ev.ws_uid}"
            msg["proces_no"] = msg["joOpId"]

        # Optional: accumulate task-window correlations for batch tail summary.
        if ev.kind == "FINALIZE_TASK" and self.write_taskavg_on_batch_finalize:
            try:
                self._accum_add(ev, corr_data, sensors)
            except Exception as e:
                if log:
                    log.error(f"[phase3v2][corr] task accum add failed: {e}", exc_info=True)

        # writes (gated). Even when persistence is disabled, clear any batch accumulators
        # on batch finalization to avoid unbounded growth.
        if not self.persist_enabled:
            if ev.kind == "FINALIZE_BATCH" and self.write_taskavg_on_batch_finalize:
                try:
                    _ = self._accum_pop_taskavg(ev)
                except Exception:
                    pass
            return CorrPersistResult(True, False, False, False, "persist_disabled")

        try:
            if self.write_legacy and ev.kind in ("FINALIZE_TASK", "FINALIZE_BATCH"):
                # legacy writer has fixed algorithm="SPEARMAN" (table keeps proces_no)
                from cassandra_utils.models.scada_correlation_matrix import ScadaCorrelationMatrix

                ScadaCorrelationMatrix.saveData(msg, corr_data)
                wrote_legacy = True
        except Exception as e:
            if log:
                log.error(f"[phase3v2][corr] legacy write failed: {e}", exc_info=True)

        try:
            if self.write_v2 and ev.kind in ("FINALIZE_TASK", "FINALIZE_BATCH"):
                from cassandra_utils.models.scada_correlation_matrix_ws_stock_batch import (
                    ScadaCorrelationMatrixWsStockBatch,
                )

                ScadaCorrelationMatrixWsStockBatch.saveData(msg, corr_data, algorithm=algo_v2, p3_1_log=log)
                wrote_v2_batch = True
        except Exception as e:
            if log:
                log.error(f"[phase3v2][corr] v2 batch write failed: {e}", exc_info=True)

        try:
            if self.write_v2 and ev.kind == "FINALIZE_WS":
                from cassandra_utils.models.scada_correlation_matrix_ws_stock_global import (
                    ScadaCorrelationMatrixWsStockGlobal,
                )

                ScadaCorrelationMatrixWsStockGlobal.saveData(msg, corr_data, algorithm=algo_v2, p3_1_log=log)
                wrote_v2_global = True
        except Exception as e:
            if log:
                log.error(f"[phase3v2][corr] v2 global write failed: {e}", exc_info=True)

        # Temporary: on batch finalize, write an extra "task average" row with
        # partition_date bumped by +taskavg_bump_ms to ensure it is the latest row.
        if (
            ev.kind == "FINALIZE_BATCH"
            and self.write_taskavg_on_batch_finalize
            and (self.write_legacy or self.write_v2)
        ):
            try:
                popped = self._accum_pop_taskavg(ev)
                if popped is not None:
                    sensors_u, sum_mat, nwin = popped
                    avg = sum_mat / float(max(1, nwin))
                    # Optional: contrast stretch for readability (temporary backend workaround).
                    avg = self._rescale_taskavg_matrix(avg, log=log)
                    avg_data = self._mat_to_corr_data(sensors_u, avg)

                    msg_avg = _build_message_from_event(ev)
                    msg_avg["crDt"] = int(ev.end_ts_ms) + int(self.taskavg_bump_ms)
                    msg_avg["measDt"] = msg_avg["crDt"]
                    msg_avg["joOpId"] = f"BATCHAVG|{ev.batch_root or '0'}"
                    msg_avg["proces_no"] = msg_avg["joOpId"]

                    # legacy (algorithm fixed)
                    if self.write_legacy:
                        from cassandra_utils.models.scada_correlation_matrix import ScadaCorrelationMatrix

                        ScadaCorrelationMatrix.saveData(msg_avg, avg_data, p3_1_log=log)

                    # v2 batch (algorithm tags the semantics)
                    if self.write_v2:
                        from cassandra_utils.models.scada_correlation_matrix_ws_stock_batch import (
                            ScadaCorrelationMatrixWsStockBatch,
                        )

                        ScadaCorrelationMatrixWsStockBatch.saveData(
                            msg_avg,
                            avg_data,
                            algorithm="SPEARMAN_TASKAVG",
                            p3_1_log=log,
                        )
            except Exception as e:
                if log:
                    log.error(f"[phase3v2][corr] batch taskavg write failed: {e}", exc_info=True)

        return CorrPersistResult(True, wrote_legacy, wrote_v2_batch, wrote_v2_global, "ok")
