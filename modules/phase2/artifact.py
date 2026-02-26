from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Phase2Artifact:
    """Phase2 artifact contract.

    This is intentionally minimal but stable. It is the interface between:
    - offline training (produces artifacts)
    - runtime inference (consumes artifacts)
    - registry routing (selects best artifact)

    Notes:
    - wsuid_token MUST be derived from canonical workstation_uid via safe_token.
    - resample_sec MUST match runtime bucketizer grid.
    """

    # routing keys
    wsuid: str
    wsuid_token: str
    resample_sec: int

    # context routing
    context_policy_id: str = "ws_stock"
    stock_key: str = "ALL"  # exact match or ALL
    op_tc: str = "ALL"      # exact match or ALL (optional)

    # model
    model_family: str = "lstm_ae"  # lstm_ae | robust_zscore | ...
    timesteps: int = 20
    sensor_cols: List[str] = None

    # preprocessing
    fillna_value: float = 0.0
    scaler_type: str = "standard"  # standard | none

    # decision
    score_type: str = "recon_mse"
    threshold: float = 0.0

    # acceptance
    accepted: bool = True
    acceptance_reason: str = ""

    # provenance
    trained_at_utc: str = ""
    n_rows_raw: int = 0
    n_rows_grid: int = 0
    n_train_seq: int = 0

    # payload pointers
    model_path: str = ""
    scaler_path: str = ""

    def __post_init__(self) -> None:
        if self.sensor_cols is None:
            self.sensor_cols = []
        if not self.trained_at_utc:
            self.trained_at_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wsuid": self.wsuid,
            "wsuid_token": self.wsuid_token,
            "resample_sec": int(self.resample_sec),
            "context_policy_id": self.context_policy_id,
            "stock_key": self.stock_key,
            "op_tc": self.op_tc,
            "model_family": self.model_family,
            "timesteps": int(self.timesteps),
            "sensor_cols": list(self.sensor_cols or []),
            "fillna_value": float(self.fillna_value),
            "scaler_type": self.scaler_type,
            "score_type": self.score_type,
            "threshold": float(self.threshold),
            "accepted": bool(self.accepted),
            "acceptance_reason": self.acceptance_reason,
            "trained_at_utc": self.trained_at_utc,
            "n_rows_raw": int(self.n_rows_raw),
            "n_rows_grid": int(self.n_rows_grid),
            "n_train_seq": int(self.n_train_seq),
            "model_path": self.model_path,
            "scaler_path": self.scaler_path,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Phase2Artifact":
        return Phase2Artifact(
            wsuid=str(d.get("wsuid") or ""),
            wsuid_token=str(d.get("wsuid_token") or ""),
            resample_sec=int(d.get("resample_sec") or 0),
            context_policy_id=str(d.get("context_policy_id") or "ws_stock"),
            stock_key=str(d.get("stock_key") or "ALL"),
            op_tc=str(d.get("op_tc") or "ALL"),
            model_family=str(d.get("model_family") or "lstm_ae"),
            timesteps=int(d.get("timesteps") or 20),
            sensor_cols=[str(x) for x in (d.get("sensor_cols") or [])],
            fillna_value=float(d.get("fillna_value") or 0.0),
            scaler_type=str(d.get("scaler_type") or "standard"),
            score_type=str(d.get("score_type") or "recon_mse"),
            threshold=float(d.get("threshold") or 0.0),
            accepted=bool(d.get("accepted") if d.get("accepted") is not None else True),
            acceptance_reason=str(d.get("acceptance_reason") or ""),
            trained_at_utc=str(d.get("trained_at_utc") or ""),
            n_rows_raw=int(d.get("n_rows_raw") or 0),
            n_rows_grid=int(d.get("n_rows_grid") or 0),
            n_train_seq=int(d.get("n_train_seq") or 0),
            model_path=str(d.get("model_path") or ""),
            scaler_path=str(d.get("scaler_path") or ""),
        )

    @staticmethod
    def load_json(path: str) -> "Phase2Artifact":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        a = Phase2Artifact.from_dict(d)
        # resolve relative payload paths relative to meta file
        base = os.path.dirname(os.path.abspath(path))
        if a.model_path and not os.path.isabs(a.model_path):
            a.model_path = os.path.join(base, a.model_path)
        if a.scaler_path and not os.path.isabs(a.scaler_path):
            a.scaler_path = os.path.join(base, a.scaler_path)
        return a
