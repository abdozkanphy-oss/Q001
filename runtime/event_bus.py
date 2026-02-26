# runtime/event_bus.py
# Central in-process queues to decouple Kafka ingest from Phase2/Phase3 workers.
#
# Design goals:
# - single producer (Kafka fanout thread) -> multiple consumers (phase2/phase3 workers)
# - bounded queues to avoid unbounded RAM under bursty streams
# - optional blocking enqueue to provide backpressure (preferred for Phase2/Phase3 correctness)

from __future__ import annotations

import queue
from typing import Optional, Tuple

from utils.config_reader import ConfigReader
from utils.keypoint_recorder import KP

cfg = ConfigReader()

# NOTE: keep defaults conservative; override via config.json if needed.
PHASE2_Q_MAX = int(getattr(cfg, "phase2_queue_max", 20000) or 20000)
PHASE3_Q_MAX = int(getattr(cfg, "phase3_queue_max", 20000) or 20000)

# backpressure knobs
ENQUEUE_BLOCK = bool(getattr(cfg, "fanout_enqueue_block", True))
ENQUEUE_TIMEOUT_SEC = float(getattr(cfg, "fanout_enqueue_timeout_sec", 1.0) or 1.0)

_PHASE2_Q: "queue.Queue[dict]" = queue.Queue(maxsize=max(1, PHASE2_Q_MAX))
_PHASE3_Q: "queue.Queue[dict]" = queue.Queue(maxsize=max(1, PHASE3_Q_MAX))


def enqueue_phase2(message: dict, *, block: Optional[bool] = None, timeout: Optional[float] = None) -> bool:
    """Returns True if enqueued, False if dropped due to timeout/full."""
    b = ENQUEUE_BLOCK if block is None else bool(block)
    t = ENQUEUE_TIMEOUT_SEC if timeout is None else float(timeout)
    try:
        _PHASE2_Q.put(message, block=b, timeout=(t if b else 0.0))
        KP.inc("event_bus.enqueue.phase2.ok", 1)
        return True
    except queue.Full:
        KP.inc("event_bus.enqueue.phase2.full", 1)
        return False


def enqueue_phase3(message: dict, *, block: Optional[bool] = None, timeout: Optional[float] = None) -> bool:
    """Returns True if enqueued, False if dropped due to timeout/full."""
    b = ENQUEUE_BLOCK if block is None else bool(block)
    t = ENQUEUE_TIMEOUT_SEC if timeout is None else float(timeout)
    try:
        _PHASE3_Q.put(message, block=b, timeout=(t if b else 0.0))
        KP.inc("event_bus.enqueue.phase3.ok", 1)
        return True
    except queue.Full:
        KP.inc("event_bus.enqueue.phase3.full", 1)
        return False


def dequeue_phase2(*, timeout: Optional[float] = None) -> dict:
    msg = _PHASE2_Q.get(timeout=timeout) if timeout is not None else _PHASE2_Q.get()
    KP.inc("event_bus.dequeue.phase2", 1)
    return msg


def dequeue_phase3(*, timeout: Optional[float] = None) -> dict:
    msg = _PHASE3_Q.get(timeout=timeout) if timeout is not None else _PHASE3_Q.get()
    KP.inc("event_bus.dequeue.phase3", 1)
    return msg


def task_done_phase2() -> None:
    _PHASE2_Q.task_done()


def task_done_phase3() -> None:
    _PHASE3_Q.task_done()


def qsize() -> Tuple[int, int]:
    return _PHASE2_Q.qsize(), _PHASE3_Q.qsize()
