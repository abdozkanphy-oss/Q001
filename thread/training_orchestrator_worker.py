from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import Optional

from utils.config_reader import ConfigReader
from utils.logger_2 import setup_logger

log = setup_logger("training_orchestrator", "logs/training_orchestrator.log")
cfg = ConfigReader()

ENABLED = bool(getattr(cfg, "training_orchestrator_enabled", False))
EVERY_SEC = float(getattr(cfg, "training_orchestrator_every_sec", 3600.0) or 3600.0)

# OUT_ONLY sweep (Phase3)
OUTONLY_SWEEP_ENABLED = bool(getattr(cfg, "training_orchestrator_outonly_sweep_enabled", True))
OUTONLY_WS_LIST = str(getattr(cfg, "training_orchestrator_outonly_ws_list", "./ws_list.txt") or "./ws_list.txt")
OUTONLY_DAYS = int(getattr(cfg, "training_orchestrator_outonly_days", 14) or 14)
OUTONLY_TOP_STOCKS = int(getattr(cfg, "training_orchestrator_outonly_top_stocks", 2) or 2)
OUTONLY_MIN_ROWS = int(getattr(cfg, "training_orchestrator_outonly_min_rows", 80) or 80)
OUTONLY_N_LAGS = int(getattr(cfg, "training_orchestrator_outonly_n_lags", 6) or 6)
OUTONLY_RESAMPLE_SEC = int(getattr(cfg, "training_orchestrator_outonly_resample_sec", 60) or 60)
OUTONLY_HORIZON_SEC = int(getattr(cfg, "training_orchestrator_outonly_horizon_sec", 60) or 60)
OUTONLY_DRY_RUN = bool(getattr(cfg, "training_orchestrator_outonly_dry_run", False))

# Phase2 unsupervised offline training (placeholder - will be expanded in "Phase2 rebuild")
PHASE2_UNSUP_ENABLED = bool(getattr(cfg, "training_orchestrator_phase2_unsup_enabled", False))
PHASE2_UNSUP_DAYS = int(getattr(cfg, "training_orchestrator_phase2_unsup_days", 14) or 14)


def _run(cmd: list[str], cwd: Optional[str] = None) -> int:
    log.info(f"[training_orchestrator] run: {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.info(p.stdout[-8000:] if p.stdout else "")
    return int(p.returncode)


def _maybe_run_outonly_sweep() -> None:
    if not OUTONLY_SWEEP_ENABLED:
        return
    if not os.path.exists(OUTONLY_WS_LIST):
        log.warning(f"[training_orchestrator] ws_list not found: {OUTONLY_WS_LIST} (skip outonly sweep)")
        return

    cmd = [
        sys.executable,
        "-m",
        "modules.offline_outonly_sweep",
        "--ws_list",
        OUTONLY_WS_LIST,
        "--days",
        str(OUTONLY_DAYS),
        "--top_stocks",
        str(OUTONLY_TOP_STOCKS),
        "--min_rows",
        str(OUTONLY_MIN_ROWS),
        "--n_lags",
        str(OUTONLY_N_LAGS),
        "--resample_sec",
        str(OUTONLY_RESAMPLE_SEC),
        "--horizon_sec",
        str(OUTONLY_HORIZON_SEC),
    ]
    if OUTONLY_DRY_RUN:
        cmd.append("--dry_run")

    rc = _run(cmd)
    if rc != 0:
        log.warning(f"[training_orchestrator] outonly sweep exited rc={rc}")


def _maybe_run_phase2_unsup() -> None:
    if not PHASE2_UNSUP_ENABLED:
        return

    cmd = [
        sys.executable,
        "-m",
        "modules.offline_phase2_unsup_trainer",
        "--days",
        str(PHASE2_UNSUP_DAYS),
    ]
    rc = _run(cmd)
    if rc != 0:
        log.warning(f"[training_orchestrator] phase2 unsup exited rc={rc}")


def execute_training_orchestrator_worker() -> None:
    log.info("[training_orchestrator] started")
    last_run = 0.0
    running = False

    while True:
        now = time.time()

        if not ENABLED:
            time.sleep(5.0)
            continue

        if running:
            # should not happen with current blocking impl; keep for future async
            time.sleep(1.0)
            continue

        if (now - last_run) < EVERY_SEC:
            time.sleep(1.0)
            continue

        try:
            running = True
            _maybe_run_outonly_sweep()
            _maybe_run_phase2_unsup()
        except Exception as e:
            log.error(f"[training_orchestrator] error: {e}", exc_info=True)
        finally:
            running = False
            last_run = time.time()

        time.sleep(1.0)
