"""thread/phase_3_v2/phase3_v2_runtime.py

Phase3V2 runtime (redesign).

Patch2:
- continuous prediction writing to scada_real_time_predictions (bucket finalize)

Patch3:
- window engine + finalize events (task/batch/ws partition)

Patch4:
- correlation computed on finalize events and written to Cassandra

Patch5:
- feature importance computed on finalize events and written to Cassandra

v2.24.1 patch1:
- predictor bucketizer is emit-once and supports opTc-windowing.
- predictor tick_flush emits last bucket on idle.
"""

from __future__ import annotations

import queue
import time
import logging

from runtime.event_bus import dequeue_phase3
from utils.config_reader import ConfigReader
from utils.keypoint_recorder import KP
from utils.logger_2 import setup_logger

from cassandra_utils.cqlengine_init import ensure_cqlengine_setup

from modules.phase3_v2.predictor_mimo_rf import Phase3V2MimoRFPredictor
from modules.phase3_v2.window_manager import WindowManager
from modules.phase3_v2.correlation_engine import Phase3V2CorrelationEngine
from modules.phase3_v2.feature_importance_engine import Phase3V2FeatureImportanceEngine

log = setup_logger(
    "phase3_v2", "logs/phase3_v2.log"
)

class Phase3V2Runtime:
    def __init__(self, cfg: ConfigReader):
        self.cfg = cfg

        # --- enable ---
        self.enabled = bool(getattr(cfg, "phase3v2_enabled", False))

        # --- prediction knobs (Patch2 + v2.24.1 patch1) ---
        self.models_dir = str(getattr(cfg, "phase3v2_models_dir", "./models/offline_mimo_rf") or "./models/offline_mimo_rf")

        self.pred_resample_sec = int(getattr(cfg, "phase3v2_pred_resample_sec", 60) or 60)
        self.pred_gap_reset_sec = int(getattr(cfg, "phase3v2_pred_gap_reset_sec", 900) or 900)
        self.pred_model_refresh_sec = int(getattr(cfg, "phase3v2_pred_model_refresh_sec", 60) or 60)
        self.pred_max_lookback_frames = int(getattr(cfg, "phase3v2_pred_max_lookback_frames", 400) or 400)
        self.pred_algorithm = str(getattr(cfg, "phase3v2_pred_algorithm", "RANDOM_FOREST") or "RANDOM_FOREST")

        self.pred_lateness_buckets = int(getattr(cfg, "phase3v2_pred_lateness_buckets", 0) or 0)
        self.pred_idle_flush_sec = int(getattr(cfg, "phase3v2_pred_idle_flush_sec", 5) or 5)
        self.pred_clear_lookback_on_optc_change = bool(
            getattr(cfg, "phase3v2_pred_clear_lookback_on_optc_change", True)
        )

        self.predictor = Phase3V2MimoRFPredictor(
            models_dir=self.models_dir,
            resample_sec_default=self.pred_resample_sec,
            gap_reset_sec=self.pred_gap_reset_sec,
            model_refresh_sec=self.pred_model_refresh_sec,
            max_lookback_frames=self.pred_max_lookback_frames,
            algorithm_name=self.pred_algorithm,
            lateness_buckets=self.pred_lateness_buckets,
            idle_flush_sec=self.pred_idle_flush_sec,
            clear_lookback_on_optc_change=self.pred_clear_lookback_on_optc_change,
        )

        # --- window knobs (Patch3) ---
        self.win_idle_finalize_sec = int(getattr(cfg, "phase3v2_win_idle_finalize_sec", 900) or 900)
        self.win_ws_partition_sec = int(getattr(cfg, "phase3v2_win_ws_partition_sec", 86400) or 86400)
        self.win_max_inmem_frames = int(getattr(cfg, "phase3v2_win_max_inmem_frames", 20000) or 20000)

        self.window_mgr = WindowManager(
            idle_finalize_sec=self.win_idle_finalize_sec,
            ws_partition_sec=self.win_ws_partition_sec,
            max_inmem_frames=self.win_max_inmem_frames,
        )

        # --- correlation on finalize (Patch4) ---
        self.corr_engine = Phase3V2CorrelationEngine(cfg)

        # --- feature importance on finalize (Patch5) ---
        self.fi_engine = Phase3V2FeatureImportanceEngine(cfg, models_dir=self.models_dir)

    def _handle_window_events(self, events):
        for ev in events:
            KP.inc(f"phase3v2.win.{ev.kind.lower()}", 1)
            KP.inc(f"phase3v2.win.{ev.kind.lower()}.reason.{ev.reason}", 1)
            log.info(
                f"[phase3v2] {ev.kind} reason={ev.reason} ws={ev.ws_uid} stock={ev.stock_key} "
                f"batch={ev.batch_id} phase={ev.phase_id} frames={ev.n_frames} "
                f"range_ms=[{ev.start_ts_ms},{ev.end_ts_ms}]"
            )

            # correlation is strictly on finalize events (never on per-message)
            if ev.kind in ("FINALIZE_TASK", "FINALIZE_BATCH", "FINALIZE_WS"):
                try:
                    res = self.corr_engine.on_window_event(ev, log=log)
                    if res.computed:
                        KP.inc("phase3v2.corr.compute", 1)
                        if res.wrote_legacy:
                            KP.inc("phase3v2.corr.write.legacy.ok", 1)
                        if res.wrote_v2_batch:
                            KP.inc("phase3v2.corr.write.v2_batch.ok", 1)
                        if res.wrote_v2_global:
                            KP.inc("phase3v2.corr.write.v2_global.ok", 1)
                    else:
                        KP.inc("phase3v2.corr.skip", 1)
                        KP.inc(f"phase3v2.corr.skip.{res.reason}", 1)
                except Exception as e:
                    KP.inc("phase3v2.corr.error", 1)
                    log.error(f"[phase3v2][corr] error: {e}", exc_info=True)

            # feature importance is also on finalize events
            if ev.kind in ("FINALIZE_TASK", "FINALIZE_BATCH", "FINALIZE_WS"):
                try:
                    res = self.fi_engine.on_window_event(ev, log=log)
                    if res.computed:
                        KP.inc("phase3v2.fi.compute", 1)
                        if res.wrote_legacy:
                            KP.inc("phase3v2.fi.write.legacy.ok", 1)
                        else:
                            KP.inc("phase3v2.fi.write.legacy.fail", 1)
                    else:
                        KP.inc("phase3v2.fi.skip", 1)
                        KP.inc(f"phase3v2.fi.skip.{res.reason}", 1)
                except Exception as e:
                    KP.inc("phase3v2.fi.error", 1)
                    log.error(f"[phase3v2][fi] error: {e}", exc_info=True)

    def _process_frames(self, frames):
        for fr in frames or []:
            # Per-frame counters for prediction outcomes
            reason = str(fr.message_meta.get("pred_reason") or "")
            wrote = bool(fr.message_meta.get("pred_wrote"))
            if wrote:
                KP.inc("phase3v2.pred.write.ok", 1)
            else:
                KP.inc("phase3v2.pred.write.skip", 1)
                if reason:
                    KP.inc(f"phase3v2.pred.skip.{reason}", 1)

            # window transitions
            events = self.window_mgr.on_frame(fr)
            if events:
                self._handle_window_events(events)

    def run_forever(self):
        if not self.enabled:
            log.info("[phase3_v2] disabled (phase3v2_enabled=false)")
            return

        # Ensure cqlengine is configured once. Many legacy model modules do not
        # call connection.setup at import time.
        try:
            ensure_cqlengine_setup(self.cfg)
        except Exception as e:
            KP.inc("phase3v2.cassandra.setup.error", 1)
            log.error(f"[phase3_v2] cassandra setup failed: {e}", exc_info=True)
            raise

        log.info(
            f"[phase3_v2] starting: models_dir={self.models_dir} "
            f"pred_rsec={self.pred_resample_sec} gap_reset_sec={self.pred_gap_reset_sec} "
            f"lateness_buckets={self.pred_lateness_buckets} idle_pred_flush_sec={self.pred_idle_flush_sec} "
            f"idle_finalize_sec={self.win_idle_finalize_sec} ws_partition_sec={self.win_ws_partition_sec} "
            f"win_max_inmem_frames={self.win_max_inmem_frames}"
        )

        last_tick = time.time()
        while True:
            try:
                msg = dequeue_phase3(timeout=1.0)
            except queue.Empty:
                msg = None
            except Exception as e:
                KP.inc("phase3v2.dequeue.error", 1)
                log.error(f"[phase3_v2] dequeue error: {e}")
                msg = None

            if msg is None:
                KP.inc("phase3v2.loop.idle", 1)
                now = time.time()
                if now - last_tick >= 2.0:
                    last_tick = now
                    # 1) flush predictor tail buckets on idle
                    try:
                        frames = self.predictor.tick_flush(log=log, cfg=self.cfg)
                        if frames:
                            KP.inc("phase3v2.frames", int(len(frames)))
                            self._process_frames(frames)
                    except Exception as e:
                        KP.inc("phase3v2.pred.tick_flush.error", 1)
                        log.error(f"[phase3_v2] predictor tick_flush error: {e}", exc_info=True)

                    # 2) finalize windows on idle
                    try:
                        events = self.window_mgr.tick()
                        if events:
                            self._handle_window_events(events)
                    except Exception as e:
                        KP.inc("phase3v2.tick.error", 1)
                        log.error(f"[phase3_v2] tick error: {e}", exc_info=True)
                continue

            KP.inc("phase3v2.msg", 1)

            try:
                frames = self.predictor.on_message(msg, log=log, cfg=self.cfg)
                KP.inc("phase3v2.frames", int(len(frames)))
                self._process_frames(frames)

            except Exception as e:
                KP.inc("phase3v2.process.error", 1)
                log.error(f"[phase3_v2] process error: {e}", exc_info=True)



def execute_phase_three_worker():
    cfg = ConfigReader()
    Phase3V2Runtime(cfg).run_forever()


if __name__ == "__main__":
    cfg = ConfigReader()
    Phase3V2Runtime(cfg).run_forever()
