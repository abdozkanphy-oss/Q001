from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np

from modules.phase3_v2.mimo_rf_registry import MimoRFArtifact, get_mimo_rf_registry
from modules.phase3_v2.types import Phase3Frame, PredictorResult
from modules.phase3_v2.utils import (
    event_ts_ms_from_msg,
    extract_meta_for_predictions,
    extract_op_tc,
    extract_sensor_values,
    extract_stock_key,
    ms_to_dt_utc,
    parse_batch_root,
    parse_phase_process_no,
    safe_float,
)


@dataclass
class _CtxState:
    ws_uid: str
    stock_key: str

    # active context ids (best-effort, updated on each message)
    batch_id: str = ""
    batch_root: str = "0"
    phase_id: str = ""  # in Phase3V2: opTc
    process_no: str = "0"

    # resample policy (effective)
    resample_sec: int = 0
    last_resample_check_ms: int = 0

    # pending buckets (emit-once)
    pending_values: Dict[int, Dict[str, float]] = None
    pending_meta: Dict[int, Dict[str, Any]] = None
    max_bucket_id_seen: Optional[int] = None
    last_emitted_bucket_id: Optional[int] = None
    last_emitted_ts_ms: int = 0

    # lookback (emitted frames)
    frames: Deque[Phase3Frame] = None

    # timing
    last_seen_wall_ms: int = 0


class Phase3V2MimoRFPredictor:
    """Continuous MIMO-RF prediction writer (Phase3V2).

    Key properties (v2.24.1 patch1):
    - Event-time bucketization with *emit-once* semantics (no rewriting old buckets).
    - Bucket merge: multiple messages in the same bucket update the same sensor map.
    - Phase/task identity uses opTc (operation task code), not joOpId.
    - Resample cadence is chosen from the available model artifact for the ws+stock(+opTc)
      context when possible; otherwise falls back to phase3v2_pred_resample_sec.
    """

    def __init__(
        self,
        *,
        models_dir: str,
        resample_sec_default: int = 60,
        gap_reset_sec: int = 900,
        model_refresh_sec: int = 60,
        max_lookback_frames: int = 400,
        algorithm_name: str = "RANDOM_FOREST",
        lateness_buckets: int = 0,
        idle_flush_sec: int = 5,
        clear_lookback_on_optc_change: bool = True,
    ):
        self.models_dir = models_dir or "./models/offline_mimo_rf"
        self.resample_sec_default = int(resample_sec_default or 60)
        self.gap_reset_sec = int(gap_reset_sec or 900)
        self.model_refresh_sec = int(model_refresh_sec or 60)
        self.max_lookback_frames = int(max_lookback_frames or 400)
        self.algorithm_name = str(algorithm_name or "RANDOM_FOREST")

        # bounded-out-of-order policy: keep last (1+lateness_buckets) buckets open
        self.lateness_buckets = int(max(0, lateness_buckets))
        self.idle_flush_sec = int(max(1, idle_flush_sec))
        self.clear_lookback_on_optc_change = bool(clear_lookback_on_optc_change)

        self._states: Dict[Tuple[str, str], _CtxState] = {}
        # joblib model cache: meta_path -> model object
        self._model_cache: Dict[str, Any] = {}

    def _get_state(self, ws_uid: str, stock_key: str) -> _CtxState:
        k = (str(ws_uid), str(stock_key))
        st = self._states.get(k)
        if st is None:
            st = _CtxState(ws_uid=str(ws_uid), stock_key=str(stock_key))
            st.pending_values = {}
            st.pending_meta = {}
            st.frames = deque(maxlen=self.max_lookback_frames)
            st.last_seen_wall_ms = int(time.time() * 1000.0)
            self._states[k] = st
        return st

    def _bucket_id(self, ts_ms: int, resample_sec: int) -> int:
        if not ts_ms:
            ts_ms = int(time.time() * 1000.0)
        step = int(resample_sec) * 1000
        return int(ts_ms // step)

    def _bucket_start_ms(self, bucket_id: int, resample_sec: int) -> int:
        step_ms = int(resample_sec) * 1000
        return int(bucket_id) * int(step_ms)

    def _maybe_refresh_resample_policy(self, st: _CtxState, *, msg: Dict[str, Any], cfg, log) -> Optional[MimoRFArtifact]:
        """Refresh the model-based resample cadence occasionally.

        Returns the "best" artifact found (any resample). We use it only to pick
        resample_sec. Prediction uses a stricter lookup by resample+opTc.
        """
        now_ms = int(time.time() * 1000.0)
        if st.last_resample_check_ms and (now_ms - int(st.last_resample_check_ms)) < int(self.model_refresh_sec) * 1000:
            return None
        st.last_resample_check_ms = now_ms

        stock_no = msg.get("produced_stock_no")
        stock_nm = msg.get("produced_stock_name")
        st_cands = [x for x in [stock_no, stock_nm, st.stock_key] if x]

        op_tc = extract_op_tc(msg)
        op_cands = [x for x in [op_tc] if x]

        reg = get_mimo_rf_registry(self.models_dir)
        # pick any resample cadence (do NOT filter by rsec here)
        art = reg.find_best(wsuid=st.ws_uid, stock_candidates=st_cands, op_candidates=op_cands, resample_sec=None)
        return art

    def _reset_stream_state(self, st: _CtxState, *, reason: str) -> None:
        # Emit-once contract: when resetting, treat past buckets as closed.
        st.pending_values = {}
        st.pending_meta = {}
        st.max_bucket_id_seen = None
        st.last_emitted_bucket_id = None
        st.last_emitted_ts_ms = 0
        st.frames.clear()

    def _emit_ready_buckets(
        self,
        st: _CtxState,
        *,
        resample_sec: int,
        cutoff_exclusive: int,
        log=None,
        cfg=None,
    ) -> List[Phase3Frame]:
        """Emit buckets with id < cutoff_exclusive, in ascending order, at most once."""
        if not st.pending_values:
            return []

        step_ms = int(resample_sec) * 1000
        out: List[Phase3Frame] = []

        # deterministic order
        for bid in sorted(list(st.pending_values.keys())):
            if bid >= int(cutoff_exclusive):
                break

            # already emitted?
            if st.last_emitted_bucket_id is not None and int(bid) <= int(st.last_emitted_bucket_id):
                # cleanup stale pending bucket
                try:
                    del st.pending_values[bid]
                    st.pending_meta.pop(bid, None)
                except Exception:
                    pass
                continue

            vals = st.pending_values.get(bid) or {}
            if not vals:
                # no sensor values; drop silently
                try:
                    del st.pending_values[bid]
                    st.pending_meta.pop(bid, None)
                except Exception:
                    pass
                continue

            meta = st.pending_meta.get(bid) or {}
            batch_id = str(meta.get("_batch_id") or st.batch_id or "")
            batch_root = str(meta.get("_batch_root") or st.batch_root or "0")
            phase_id = str(meta.get("_phase_id") or st.phase_id or "0")
            process_no = str(meta.get("_process_no") or st.process_no or "0")

            ts_ms = int(bid) * int(step_ms)
            ts_utc = ms_to_dt_utc(ts_ms)

            frame = Phase3Frame(
                ws_uid=st.ws_uid,
                stock_key=st.stock_key,
                ts_utc=ts_utc,
                ts_ms=ts_ms,
                batch_id=batch_id,
                batch_root=batch_root,
                phase_id=phase_id,
                message_meta={
                    "process_no": process_no,
                    "resample_sec": int(resample_sec),
                },
                sensor_values=dict(vals),
            )

            # enrich with per-bucket meta (best-effort)
            try:
                frame.message_meta.update(meta)
                # legacy expectations
                frame.message_meta["refNo"] = str(batch_root or "0")
                frame.message_meta["joOpId"] = str(phase_id or "0")
                frame.message_meta["proces_no"] = str(process_no or "0")
            except Exception:
                pass

            # lookback: avoid cross-opTc leakage unless explicitly allowed
            if self.clear_lookback_on_optc_change and st.frames:
                try:
                    if str(st.frames[-1].phase_id or "") != str(frame.phase_id or ""):
                        st.frames.clear()
                except Exception:
                    pass

            st.frames.append(frame)
            out.append(frame)

            st.last_emitted_bucket_id = int(bid)
            st.last_emitted_ts_ms = int(ts_ms)

            # cleanup pending
            try:
                del st.pending_values[bid]
                st.pending_meta.pop(bid, None)
            except Exception:
                pass

        # Run prediction writes for emitted frames
        for fr in out:
            try:
                res = self._write_prediction(frame=fr, st=st, log=log, cfg=cfg)
                fr.message_meta["pred_wrote"] = bool(res.wrote)
                fr.message_meta["pred_reason"] = str(res.reason)
                if res.model_id:
                    fr.message_meta["pred_model"] = str(res.model_id)
            except Exception as e:
                if log:
                    log.error(f"[phase3v2] prediction loop failed: {e}", exc_info=True)
                fr.message_meta["pred_wrote"] = False
                fr.message_meta["pred_reason"] = "predict_loop_error"

        return out

    def _select_artifact_for_prediction(
        self,
        *,
        ws_uid: str,
        stock_key: str,
        meta: Dict[str, Any],
        op_tc: str,
        resample_sec: int,
    ) -> Optional[MimoRFArtifact]:
        reg = get_mimo_rf_registry(self.models_dir)

        stock_candidates: List[str] = []
        for c in [meta.get("produced_stock_no"), meta.get("produced_stock_name"), stock_key]:
            if c is None:
                continue
            s = str(c).strip()
            if s and s not in stock_candidates:
                stock_candidates.append(s)

        op_candidates: List[str] = []
        for c in [op_tc]:
            if c is None:
                continue
            s = str(c).strip()
            if s and s not in op_candidates:
                op_candidates.append(s)

        return reg.find_best(
            wsuid=str(ws_uid),
            stock_candidates=stock_candidates,
            op_candidates=op_candidates,
            resample_sec=int(resample_sec) if int(resample_sec) > 0 else None,
        )

    def _load_model_cached(self, art: MimoRFArtifact, log=None) -> Optional[Any]:
        if art is None:
            return None
        if art.meta_path in self._model_cache:
            return self._model_cache.get(art.meta_path)
        try:
            model = joblib.load(art.model_path)
            self._model_cache[art.meta_path] = model
            return model
        except Exception as e:
            if log:
                log.error(f"[phase3v2] failed to load MIMO RF model: {art.model_path} err={e}")
            return None

    def _build_feature_vector(self, frames: Deque[Phase3Frame], art: MimoRFArtifact) -> Optional[np.ndarray]:
        if art is None:
            return None
        if not art.feature_names or int(art.n_lags) <= 0:
            return None
        if len(frames) < int(art.n_lags):
            return None

        frames_list = list(frames)
        x_vals: List[float] = []

        for fn in art.feature_names:
            # format: "sensor__lag_k"
            try:
                sensor, lag_s = fn.rsplit("__lag_", 1)
                lag = int(lag_s)
            except Exception:
                x_vals.append(0.0)
                continue

            if lag <= 0 or lag > len(frames_list):
                x_vals.append(0.0)
                continue

            v = frames_list[-lag].sensor_values.get(sensor)
            x_vals.append(safe_float(v, default=0.0))

        X = np.asarray(x_vals, dtype="float32").reshape(1, -1)
        return X

    def _write_prediction(self, *, frame: Phase3Frame, st: _CtxState, log=None, cfg=None) -> PredictorResult:
        # global derived-output persistence gate (compute is allowed; writes are not)
        persist_enabled = True
        try:
            persist_enabled = bool(getattr(cfg, "phase3_derived_persist_enabled", True)) if cfg is not None else True
        except Exception:
            persist_enabled = True

        op_tc = str(frame.phase_id or frame.message_meta.get("opTc") or "")
        resample_sec = int(frame.message_meta.get("resample_sec") or st.resample_sec or self.resample_sec_default)

        art = self._select_artifact_for_prediction(
            ws_uid=st.ws_uid,
            stock_key=st.stock_key,
            meta=frame.message_meta,
            op_tc=op_tc,
            resample_sec=resample_sec,
        )
        if art is None:
            return PredictorResult(wrote=False, reason="no_model")

        model = self._load_model_cached(art, log=log)
        if model is None:
            return PredictorResult(wrote=False, reason="model_load_failed", model_id=art.meta_path)

        X = self._build_feature_vector(st.frames, art)
        if X is None:
            return PredictorResult(wrote=False, reason="insufficient_lookback", model_id=art.meta_path)

        # predict
        try:
            y = model.predict(X)
            y = np.asarray(y, dtype="float64")
            if y.ndim == 1:
                y = y.reshape(1, -1)
            if y.shape[0] != 1:
                return PredictorResult(wrote=False, reason="predict_shape", model_id=art.meta_path)
            preds = y[0].tolist()
        except Exception as e:
            if log:
                log.error(f"[phase3v2] model.predict failed: {e}", exc_info=True)
            return PredictorResult(wrote=False, reason="predict_error", model_id=art.meta_path)

        targets = art.targets or []
        if len(targets) != len(preds):
            return PredictorResult(wrote=False, reason="target_mismatch", model_id=art.meta_path)

        cur_vals = st.frames[-1].sensor_values if st.frames else frame.sensor_values

        # Baseline "mean" for Cassandra payloads.
        # Keep this intentionally simple and stable: rolling mean over recent emitted frames.
        try:
            mean_window_frames = int(getattr(cfg, "phase3v2_pred_mean_window_frames", 30) or 30) if cfg is not None else 30
        except Exception:
            mean_window_frames = 30
        mean_window_frames = int(max(1, min(1000, mean_window_frames)))

        def _rolling_mean(sensor_name: str) -> float:
            try:
                frs = list(st.frames)[-mean_window_frames:] if st.frames else []
            except Exception:
                frs = []
            vals: List[float] = []
            for fr in frs:
                try:
                    v = (fr.sensor_values or {}).get(sensor_name)
                    fv = float(v)
                    if np.isfinite(fv):
                        vals.append(float(fv))
                except Exception:
                    continue
            if not vals:
                return float(safe_float(cur_vals.get(sensor_name), default=0.0))
            try:
                return float(np.mean(np.asarray(vals, dtype="float64")))
            except Exception:
                return float(safe_float(cur_vals.get(sensor_name), default=0.0))

        # build payloads
        input_payload: Dict[str, Dict[str, float]] = {}
        for s in (art.sensors_used or []):
            av = safe_float(cur_vals.get(s), default=0.0)
            mv = _rolling_mean(str(s))
            input_payload[str(s)] = {"actual": float(av), "mean": float(mv), "predicted": float(av)}

        output_payload: Dict[str, Dict[str, float]] = {}
        for t, pv in zip(targets, preds):
            av = safe_float(cur_vals.get(t), default=0.0)
            mv = _rolling_mean(str(t))
            output_payload[str(t)] = {
                "actual": float(av),
                "mean": float(mv),
                "predicted": safe_float(pv, default=0.0),
            }

        # meta for saveData
        meta = extract_meta_for_predictions(
            frame.message_meta,
            batch_root=str(frame.batch_root or "0"),
            process_no=str(frame.message_meta.get("process_no") or "0"),
            ts_utc=frame.ts_utc,
        )
        meta["_ws_uid"] = st.ws_uid
        meta["_stock_key"] = st.stock_key
        meta["_model_meta"] = art.meta_path
        meta["_model_tset"] = art.tset
        meta["_mimo_hsec"] = int(art.horizon_sec)
        meta["_mimo_rsec"] = int(art.resample_sec)
        meta["_mimo_nlag"] = int(art.n_lags)

        if not persist_enabled:
            return PredictorResult(wrote=False, reason="persist_disabled", model_id=art.meta_path)

        # write
        try:
            from cassandra_utils.models.scada_real_time_predictions import ScadaRealTimePredictions

            key = str(meta.get("process_no") or "0")
            ScadaRealTimePredictions.saveData(
                key=key,
                now_ts=frame.ts_utc,
                algorithm=self.algorithm_name,
                input_payload=input_payload,
                output_payload=output_payload,
                meta=meta,
                p3_1_log=log,
            )
            return PredictorResult(wrote=True, reason="ok", model_id=art.meta_path)
        except Exception as e:
            if log:
                log.error(f"[phase3v2] Cassandra write scada_real_time_predictions failed: {e}", exc_info=True)
            return PredictorResult(wrote=False, reason="write_error", model_id=art.meta_path)

    def on_message(self, msg: Dict[str, Any], *, log=None, cfg=None) -> List[Phase3Frame]:
        """Consume one Stage0 message. Returns 0..N emitted frames."""

        ws_uid = str(msg.get("_workstation_uid") or "")
        if not ws_uid:
            return []

        stock_key = extract_stock_key(msg)
        st = self._get_state(ws_uid, stock_key)

        st.last_seen_wall_ms = int(time.time() * 1000.0)

        # ids
        batch_id_raw = msg.get("_batch_id") or msg.get("batch_id") or msg.get("refNo")
        batch_id, batch_root = parse_batch_root(batch_id_raw)

        op_tc = extract_op_tc(msg)
        phase_id = str(op_tc or "")
        _phase_id_norm, process_no = parse_phase_process_no(phase_id)

        # resample cadence: model-based if available, else default from config
        default_rsec = int(getattr(cfg, "phase3v2_pred_resample_sec", 0) or 0) if cfg is not None else 0
        if default_rsec <= 0:
            default_rsec = self.resample_sec_default

        prev_rsec = int(st.resample_sec or 0)
        try:
            art_any = self._maybe_refresh_resample_policy(st, msg=msg, cfg=cfg, log=log)
        except Exception:
            art_any = None

        # Keep the previously chosen cadence unless we actually refreshed and found a model.
        effective_rsec = int(prev_rsec if prev_rsec > 0 else default_rsec)
        if art_any is not None:
            try:
                cand_rsec = int(getattr(art_any, "resample_sec", 0) or 0)
                if cand_rsec > 0:
                    effective_rsec = int(cand_rsec)
            except Exception:
                pass

        # if cadence changes, reset state to avoid mixed-bucket semantics
        if prev_rsec and int(prev_rsec) != int(effective_rsec):
            self._reset_stream_state(st, reason="resample_change")

        st.resample_sec = int(effective_rsec)

        ts_ms = int(event_ts_ms_from_msg(msg) or 0)
        bid = self._bucket_id(ts_ms, st.resample_sec)

        # gap reset: large forward bucket jump
        if st.max_bucket_id_seen is not None:
            try:
                gap_buckets = int(bid) - int(st.max_bucket_id_seen)
                gap_thresh = max(1, int(self.gap_reset_sec // max(1, st.resample_sec)))
                if gap_buckets > gap_thresh:
                    # close history and avoid bridging missing data
                    st.frames.clear()
                    st.pending_values = {}
                    st.pending_meta = {}
                    st.last_emitted_bucket_id = int(bid) - 1
            except Exception:
                pass

        # detect context changes
        batch_changed = bool(st.batch_id and batch_id and st.batch_id != batch_id)

        # If batch changes, flush everything we have and reset; otherwise, keep streaming.
        frames_out: List[Phase3Frame] = []
        if batch_changed:
            if st.max_bucket_id_seen is not None:
                frames_out.extend(
                    self._emit_ready_buckets(
                        st,
                        resample_sec=st.resample_sec,
                        cutoff_exclusive=int(st.max_bucket_id_seen) + 1,
                        log=log,
                        cfg=cfg,
                    )
                )
            # reset per-batch state (but keep resample cadence)
            st.pending_values = {}
            st.pending_meta = {}
            st.max_bucket_id_seen = None
            st.last_emitted_bucket_id = None
            st.last_emitted_ts_ms = 0
            st.frames.clear()

        # update active context
        st.batch_id = str(batch_id or st.batch_id or "")
        st.batch_root = str(batch_root or st.batch_root or "0")
        st.phase_id = str(phase_id or st.phase_id or "")
        st.process_no = str(process_no or st.process_no or "0")

        # late drop: do not reopen emitted buckets
        if st.last_emitted_bucket_id is not None and int(bid) <= int(st.last_emitted_bucket_id):
            return frames_out

        # update max seen
        st.max_bucket_id_seen = int(bid) if st.max_bucket_id_seen is None else max(int(st.max_bucket_id_seen), int(bid))

        # accumulate
        vals = extract_sensor_values(msg)
        if vals:
            bucket_map = st.pending_values.get(int(bid))
            if bucket_map is None:
                bucket_map = {}
                st.pending_values[int(bid)] = bucket_map
            bucket_map.update(vals)

        # minimal meta for writers; store per-bucket (latest wins)
        meta_common: Dict[str, Any] = {
            "customer": msg.get("customer") or msg.get("cust"),
            "plId": msg.get("plId") or msg.get("plant_id"),
            "wcNm": msg.get("wcNm") or msg.get("work_center_name"),
            "wcNo": msg.get("wcNo")
            or msg.get("work_center_no")
            or msg.get("wcId")
            or msg.get("work_center_id"),
            "wsNm": msg.get("wsNm") or msg.get("work_station_name"),
            "wsNo": msg.get("wsNo")
            or msg.get("work_station_no")
            or msg.get("wsId")
            or msg.get("work_station_id"),
            "opNm": msg.get("opNm") or msg.get("operator_name"),
            "opNo": msg.get("opNo") or msg.get("operator_no"),
            "produced_stock_no": msg.get("produced_stock_no")
            or msg.get("output_stock_no")
            or msg.get("stNo"),
            "produced_stock_name": msg.get("produced_stock_name")
            or msg.get("output_stock_name")
            or msg.get("stNm"),
            "joRef": msg.get("joRef") or msg.get("job_order_reference_no"),
            "proces_no": msg.get("proces_no") or msg.get("process_no"),
            "opTc": op_tc,
            # internal helpers for bucket finalize
            "_batch_id": st.batch_id,
            "_batch_root": st.batch_root,
            "_phase_id": st.phase_id,
            "_process_no": st.process_no,
        }

        pl_stno = meta_common.get("produced_stock_no")
        pl_stnm = meta_common.get("produced_stock_name")
        if pl_stno or pl_stnm:
            meta_common["prodList"] = [{"stNo": pl_stno, "stNm": pl_stnm}]

        st.pending_meta[int(bid)] = dict(meta_common)

        # emit policy: keep last (1+lateness_buckets) buckets open
        max_seen = int(st.max_bucket_id_seen or bid)
        cutoff_exclusive = int(max_seen) - int(self.lateness_buckets)
        # Always keep the current bucket open => cutoff is exclusive and we emit bid < cutoff.
        frames_out.extend(
            self._emit_ready_buckets(
                st,
                resample_sec=st.resample_sec,
                cutoff_exclusive=int(cutoff_exclusive),
                log=log,
                cfg=cfg,
            )
        )

        return frames_out

    def tick_flush(self, *, log=None, cfg=None) -> List[Phase3Frame]:
        """Flush pending buckets for idle streams.

        This is required to ensure the last bucket of a stream gets emitted when
        no new bucket boundary arrives.

        Policy: if no messages seen for idle_flush_sec, emit all pending buckets.
        """
        now_ms = int(time.time() * 1000.0)
        idle_flush_sec = self.idle_flush_sec
        try:
            if cfg is not None:
                idle_flush_sec = int(getattr(cfg, "phase3v2_pred_idle_flush_sec", idle_flush_sec) or idle_flush_sec)
        except Exception:
            pass

        out: List[Phase3Frame] = []
        for st in list(self._states.values()):
            if not st.pending_values:
                continue
            if now_ms - int(st.last_seen_wall_ms or now_ms) < int(max(1, idle_flush_sec)) * 1000:
                continue
            if st.max_bucket_id_seen is None:
                continue

            out.extend(
                self._emit_ready_buckets(
                    st,
                    resample_sec=int(st.resample_sec or self.resample_sec_default),
                    cutoff_exclusive=int(st.max_bucket_id_seen) + 1,
                    log=log,
                    cfg=cfg,
                )
            )

        return out
