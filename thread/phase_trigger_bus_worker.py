from __future__ import annotations

import time

from thread.phase_trigger_bus import gc_expired, stats
from utils.config_reader import ConfigReader
from utils.logger_2 import setup_logger

log = setup_logger("trigger_bus", "logs/trigger_bus.log")
cfg = ConfigReader()

GC_EVERY_SEC = float(getattr(cfg, "phase_trigger_bus_gc_every_sec", 5.0) or 5.0)
LOG_EVERY_SEC = float(getattr(cfg, "phase_trigger_bus_log_every_sec", 30.0) or 30.0)


def execute_phase_trigger_bus_worker() -> None:
    log.info("[trigger_bus] started")
    last_log = 0.0
    while True:
        now = time.time()
        removed = gc_expired(now)
        if (now - last_log) >= LOG_EVERY_SEC:
            s = stats()
            log.info(f"[trigger_bus] stats={s} removed={removed}")
            last_log = now
        time.sleep(GC_EVERY_SEC)
