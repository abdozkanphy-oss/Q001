from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Phase3Frame:
    """A single resampled frame produced from streaming messages.

    - ts_utc: bucket timestamp in UTC
    - sensor_values: wide mapping sensor_name -> value
    - batch_id: Stage0 batch id (e.g. "REFNO:009927")
    - batch_root: normalized batch identifier used for Cassandra PK (e.g. "009927" or "0")
    - phase_id: task identifier (e.g. "PID_894190")
    """

    ws_uid: str
    stock_key: str
    ts_utc: datetime
    ts_ms: int

    batch_id: str
    batch_root: str
    phase_id: str

    message_meta: Dict[str, Any]
    sensor_values: Dict[str, float]


@dataclass
class PredictorResult:
    wrote: bool
    reason: str
    model_id: Optional[str] = None
