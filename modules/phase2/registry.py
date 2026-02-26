from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib

from modules.phase2.artifact import Phase2Artifact


@dataclass
class Phase2LoadResult:
    artifact: Phase2Artifact
    model: object


class Phase2Registry:
    """Loads Phase2 artifacts and routes them for runtime scoring.

    Compatibility:
    - Supports legacy Phase2Meta format produced by modules/offline_phase2_unsup_trainer.py
      (keys: wsuid, stock_key, op_tc, resample_sec, timesteps, sensor_cols, threshold, model_path, scaler_path).
    - Supports Phase2Artifact json format.

    Discovery:
    - Scans one or more directories for files:
        * "*__meta.json" (legacy)
        * "*__phase2_artifact.json" (new)
        * "phase2_artifact.json" (single-file fallback)
    """

    def __init__(self, *, scan_dirs: List[str]) -> None:
        self.scan_dirs = [d for d in (scan_dirs or []) if str(d).strip()]
        self._artifact_cache: Dict[str, Phase2Artifact] = {}
        self._model_cache: Dict[str, object] = {}
        self._scaler_cache: Dict[str, object] = {}

    @staticmethod
    def _canon_stock(s: str) -> str:
        s = (s or "ALL").strip()
        return s if s else "ALL"

    @staticmethod
    def _canon_op_tc(s: str) -> str:
        s = (s or "ALL").strip()
        return s if s else "ALL"

    @staticmethod
    def _looks_repo_root_relative(p: str) -> bool:
        """
        Detect paths that are intended to be relative to repo root, not to the meta file directory.
        Examples:
          - ".\\models\\phase2models\\offline_unsup\\X"
          - "models\\phase2models\\offline_unsup\\X"
          - "./models/phase2models/offline_unsup/X"
        """
        if not p:
            return False
        p2 = os.path.normpath(p)
        # Normalize possible mixed separators
        p2 = p2.replace("/", os.sep)
        return (
            p2.startswith(f".{os.sep}models{os.sep}")
            or p2.startswith(f"models{os.sep}")
        )

    def _discover_meta_files(self) -> List[str]:
        out: List[str] = []
        for d in self.scan_dirs:
            d = os.path.abspath(d)
            if not os.path.isdir(d):
                continue
            out += glob.glob(os.path.join(d, "**", "*__phase2_artifact.json"), recursive=True)
            out += glob.glob(os.path.join(d, "**", "phase2_artifact.json"), recursive=True)
            out += glob.glob(os.path.join(d, "**", "*__meta.json"), recursive=True)
        return sorted(set(out))

    def refresh(self) -> int:
        files = self._discover_meta_files()
        loaded = 0
        for p in files:
            try:
                a = self._load_artifact_meta(p)
                key = self._artifact_key(a)
                self._artifact_cache[key] = a
                loaded += 1
            except Exception:
                continue
        return loaded

    def _load_artifact_meta(self, meta_path: str) -> Phase2Artifact:
        with open(meta_path, "r", encoding="utf-8") as f:
            d = json.load(f)

        # New format
        if "wsuid_token" in d and "model_family" in d:
            return Phase2Artifact.load_json(meta_path)

        # Legacy Phase2Meta
        wsuid = str(d.get("wsuid") or "")
        stock_key = str(d.get("stock_key") or "ALL")
        op_tc = str(d.get("op_tc") or "ALL")
        resample_sec = int(d.get("resample_sec") or 0)
        timesteps = int(d.get("timesteps") or 20)
        sensor_cols = [str(x) for x in (d.get("sensor_cols") or [])]
        threshold = float(d.get("threshold") or 0.0)
        model_path = str(d.get("model_path") or "")
        scaler_path = str(d.get("scaler_path") or "")

        # Resolve payloads relative to meta ONLY when they truly are meta-relative.
        # If the legacy meta stored repo-root relative paths (e.g. ".\\models\\..."), do NOT join with meta dir.
        base = os.path.dirname(os.path.abspath(meta_path))

        if model_path and (not os.path.isabs(model_path)) and (not self._looks_repo_root_relative(model_path)):
            model_path = os.path.join(base, model_path)

        if scaler_path and (not os.path.isabs(scaler_path)) and (not self._looks_repo_root_relative(scaler_path)):
            scaler_path = os.path.join(base, scaler_path)

        # wsuid_token: reuse safe_token; prefer modules.model_registry (dependency-light), fallback for older trees.
        try:
            from modules.model_registry import safe_token
        except Exception:  # pragma: no cover
            from modules.offline_outonly_trainer import safe_token

        a = Phase2Artifact(
            wsuid=wsuid,
            wsuid_token=safe_token(wsuid),
            resample_sec=resample_sec,
            context_policy_id="ws_stock",
            stock_key=self._canon_stock(stock_key),
            op_tc=self._canon_op_tc(op_tc),
            model_family="lstm_ae",
            timesteps=timesteps,
            sensor_cols=sensor_cols,
            fillna_value=0.0,
            scaler_type="standard",
            score_type="recon_mse",
            threshold=threshold,
            accepted=True,
            acceptance_reason="legacy_meta",
            trained_at_utc=str(d.get("trained_at_utc") or ""),
            n_rows_raw=int(d.get("n_rows_raw") or 0),
            n_rows_grid=int(d.get("n_rows_grid") or 0),
            n_train_seq=int(d.get("n_train_seq") or 0),
            model_path=model_path,
            scaler_path=scaler_path,
        )
        return a

    def _artifact_key(self, a: Phase2Artifact) -> str:
        return "|".join(
            [
                a.wsuid_token,
                str(int(a.resample_sec)),
                a.context_policy_id,
                self._canon_stock(a.stock_key),
                self._canon_op_tc(a.op_tc),
                a.model_family,
            ]
        )

    def _resolve_path(self, p: Optional[str]) -> str:
        """
        Resolve artifact paths robustly.

        Supports:
        - absolute paths
        - paths relative to any scan_dir
        - repo-root-ish relative paths like ".\\models\\phase2models\\offline_unsup\\X"
        - prevents double-join by selecting the first existing candidate
        """
        if not p:
            raise ValueError("empty path")
        p_norm = os.path.normpath(str(p))

        # Absolute -> use directly
        if os.path.isabs(p_norm) and os.path.exists(p_norm):
            return os.path.abspath(p_norm)

        # If it already exists relative to CWD (common on Windows when launched from repo root)
        if os.path.exists(p_norm):
            return os.path.abspath(p_norm)

        # Try relative to each scan_dir
        for d in getattr(self, "scan_dirs", []) or []:
            d_abs = os.path.abspath(d)
            cand = os.path.abspath(os.path.join(d_abs, p_norm))
            if os.path.exists(cand):
                return cand

            # Also handle when p already includes "models/phase2models/offline_unsup/..."
            # and scan_dir itself is ".../models/phase2models/offline_unsup"
            # -> joining would double. Instead try basename join as a fallback.
            cand2 = os.path.abspath(os.path.join(d_abs, os.path.basename(p_norm)))
            if os.path.exists(cand2):
                return cand2

        # Give a helpful error with candidates
        raise ValueError(
            f"File not found after resolve: path={p_norm} scan_dirs={getattr(self, 'scan_dirs', None)}"
        )

    def find_best(
        self,
        *,
        wsuid_token: str,
        resample_sec: int,
        context_policy_id: str,
        stock_key: str,
        op_tc: str = "ALL",
        model_family: Optional[str] = None,
        prefer_accepted: bool = True,
    ) -> Optional[Phase2Artifact]:
        wsuid_token = str(wsuid_token)
        rsec = int(resample_sec)
        cp = str(context_policy_id or "ws_stock")
        sk = self._canon_stock(stock_key)
        ot = self._canon_op_tc(op_tc)

        candidates: List[Phase2Artifact] = []
        for a in self._artifact_cache.values():
            if a.wsuid_token != wsuid_token:
                continue
            if int(a.resample_sec) != rsec:
                continue
            if str(a.context_policy_id or "") != cp:
                continue
            if model_family and str(a.model_family) != str(model_family):
                continue
            if self._canon_stock(a.stock_key) not in (sk, "ALL"):
                continue
            if self._canon_op_tc(a.op_tc) not in (ot, "ALL"):
                continue
            candidates.append(a)

        if not candidates:
            return None

        def rank(a: Phase2Artifact) -> Tuple[int, int, int, str]:
            acc = 1 if bool(a.accepted) else 0
            stock_exact = 1 if self._canon_stock(a.stock_key) == sk else 0
            op_exact = 1 if self._canon_op_tc(a.op_tc) == ot else 0
            return (acc, stock_exact, op_exact, str(a.trained_at_utc))

        candidates.sort(key=rank, reverse=True)
        if prefer_accepted:
            for a in candidates:
                if a.accepted:
                    return a
        return candidates[0]

    def load_model(self, a: Phase2Artifact) -> Phase2LoadResult:
        key = self._artifact_key(a)
        if key in self._model_cache:
            return Phase2LoadResult(artifact=a, model=self._model_cache[key])

        if a.model_family == "lstm_ae":
            from tensorflow.keras.models import load_model  # type: ignore
            if not a.model_path:
                raise ValueError("artifact missing model_path")

            model_path = self._resolve_path(a.model_path)
            m = load_model(model_path)

            self._model_cache[key] = m
            return Phase2LoadResult(artifact=a, model=m)

        raise ValueError(f"unsupported model_family={a.model_family}")

    def load_scaler(self, a: Phase2Artifact):
        key = self._artifact_key(a)
        if key in self._scaler_cache:
            return self._scaler_cache[key]
        if a.scaler_type in ("none", ""):
            self._scaler_cache[key] = None
            return None
        if not a.scaler_path:
            self._scaler_cache[key] = None
            return None

        scaler_path = self._resolve_path(a.scaler_path)
        sc = joblib.load(scaler_path)

        self._scaler_cache[key] = sc
        return sc