# modules/model_registry.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_token(x: str, default: str = "UNKNOWN") -> str:
    if x is None:
        return default
    s = str(x).strip()
    if not s or s.lower() == "none":
        return default
    return _SAFE_TOKEN_RE.sub("_", s)


# Filename produced by offline_outonly_trainer.py (v2.17+):
# WSUID_<wsuidtoken>_ST_<stock>_OPTC_<op>__OUTONLY__TGT_<tgt>__HSEC_<sec>__RSEC_<sec>__FM_<UNI|MV>__meta.json
# Legacy variants may omit __FM_* and/or __RSEC_*
_META_RE = re.compile(
    r"^WSUID_(?P<wsuid>.+?)_ST_(?P<st>.+?)_OPTC_(?P<op>.+?)__OUTONLY__TGT_(?P<tgt>.+?)__HSEC_(?P<hsec>\d+)(?:__RSEC_(?P<rsec>\d+))?(?:__FM_(?P<fm>[A-Z0-9]+))?__meta\.json$"
)


@dataclass(frozen=True)
class OutOnlyArtifact:
    wsuid_token: str
    stock_tag: str
    op_tag: str
    target_tag: str
    horizon_sec: int
    resample_sec: int
    model_path: str
    meta_path: str

    # evaluation
    mae: Optional[float]
    baseline_mae_lag0: Optional[float]
    lift_vs_lag0: Optional[float]
    accepted: bool
    eval_mode: str

    n_test: int
    n_train: int
    trained_at_utc: str

    # feature space
    feature_names: List[str]
    feature_mode: str


class ModelRegistry:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._outonly: List[OutOnlyArtifact] = []
        self._indexed: bool = False

    def refresh(self) -> None:
        self._outonly = []
        if not os.path.isdir(self.model_dir):
            self._indexed = True
            return

        for fn in os.listdir(self.model_dir):
            m = _META_RE.match(fn)
            if not m:
                continue

            meta_path = os.path.join(self.model_dir, fn)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
            except Exception:
                continue

            # model path: prefer meta field; fallback derived from meta filename
            model_path = meta.get("model_path")
            if not model_path:
                # replace __meta.json with __ALG_RF.pkl (current trainer)
                model_path = os.path.join(self.model_dir, fn.replace("__meta.json", "__ALG_RF.pkl"))

            metrics = meta.get("metrics") or {}

            def _f(x):
                try:
                    return float(x)
                except Exception:
                    return None

            mae = _f(metrics.get("model_mae"))
            if mae is None:
                mae = _f(metrics.get("mae"))

            baseline = _f(metrics.get("baseline_mae_lag0"))
            if baseline is None:
                # legacy field (lag1 baseline)
                baseline = _f(metrics.get("baseline_mae_lag1"))

            lift = _f(metrics.get("lift_vs_lag0"))
            if lift is None and mae is not None and baseline is not None:
                lift = float(baseline - mae)

            accepted = metrics.get("accepted")
            if accepted is None:
                # legacy: treat as accepted to preserve previous behavior
                accepted = True
            accepted = bool(accepted)

            eval_mode = str(metrics.get("eval_mode") or meta.get("eval_mode") or "")

            n_test = int(metrics.get("n_test") or 0)
            n_train = int(metrics.get("n_train") or 0)
            trained_at = str(meta.get("trained_at_utc") or "")

            feature_names = metrics.get("feature_names") or meta.get("feature_names") or []
            if not isinstance(feature_names, list):
                feature_names = []

            resample_sec = int(meta.get("resample_sec") or (m.group("rsec") or 60) or 60)

            fm = str(meta.get("feature_mode") or (m.group("fm") or "UNI") or "UNI").upper()

            art = OutOnlyArtifact(
                wsuid_token=str(m.group("wsuid")),
                stock_tag=str(m.group("st")),
                op_tag=str(m.group("op")),
                target_tag=str(m.group("tgt")),
                horizon_sec=int(m.group("hsec")),
                resample_sec=int(resample_sec),
                model_path=str(model_path),
                meta_path=str(meta_path),
                mae=mae,
                baseline_mae_lag0=baseline,
                lift_vs_lag0=lift,
                accepted=accepted,
                eval_mode=eval_mode,
                n_test=n_test,
                n_train=n_train,
                trained_at_utc=trained_at,
                feature_names=[str(x) for x in feature_names],
                feature_mode=fm,
            )

            self._outonly.append(art)

        self._indexed = True

    def list_outonly(self) -> List[OutOnlyArtifact]:
        if not self._indexed:
            self.refresh()
        return list(self._outonly)

    def find_best_outonly(
        self,
        wsuid_token: str,
        stock: str,
        op_tc: str,
        target: str,
        horizon_sec: int,
        resample_sec: Optional[int] = None,
        min_test: int = 20,
        require_accepted: bool = True,
    ) -> Optional[OutOnlyArtifact]:
        """
        Matching strategy (strict → fallback):
          - wsuid_token must match
          - horizon_sec must match
          - target token must match (safe_token)
          - optional resample_sec constraint (recommended)
          - try (stock, op_tc) exact, then op=ALL, then stock=ALL, then both ALL

        Selection:
          - prefer accepted models (if require_accepted=True)
          - among candidates with enough test: lowest MAE
          - else fallback: any MAE, else largest n_test
        """
        if not self._indexed:
            self.refresh()

        wsuid_token = safe_token(wsuid_token)
        stock_tag = safe_token(stock, default="ALL")
        op_tag = safe_token(op_tc, default="ALL")
        tgt_tag = safe_token(target)

        base = [
            a for a in self._outonly
            if a.wsuid_token == wsuid_token
            and a.horizon_sec == int(horizon_sec)
            and a.target_tag == tgt_tag
            and (resample_sec is None or int(getattr(a, "resample_sec", 0) or 0) == int(resample_sec))
        ]
        if not base:
            return None

        if require_accepted:
            base_acc = [a for a in base if bool(getattr(a, "accepted", True))]
            if base_acc:
                base = base_acc

        ladders = [
            (stock_tag, op_tag),
            (stock_tag, "ALL"),
            ("ALL", op_tag),
            ("ALL", "ALL"),
        ]

        def _best(cands: List[OutOnlyArtifact]) -> Optional[OutOnlyArtifact]:
            good = [a for a in cands if a.mae is not None and a.n_test >= int(min_test)]
            if good:
                return sorted(good, key=lambda a: (float(a.mae), -int(a.n_test or 0)))[0]
            any_mae = [a for a in cands if a.mae is not None]
            if any_mae:
                return sorted(any_mae, key=lambda a: (float(a.mae), -int(a.n_test or 0)))[0]
            return sorted(cands, key=lambda a: int(a.n_test or 0), reverse=True)[0] if cands else None

        for st, op in ladders:
            cands = [a for a in base if a.stock_tag == st and a.op_tag == op]
            if not cands:
                continue
            best = _best(cands)
            if best is not None:
                return best

        return None


# singleton registry cache per dir
_REGISTRY_BY_DIR: Dict[str, ModelRegistry] = {}


def get_outonly_registry(model_dir: str) -> ModelRegistry:
    model_dir = model_dir or "./models/offline_outonly"
    reg = _REGISTRY_BY_DIR.get(model_dir)
    if reg is None:
        reg = ModelRegistry(model_dir)
        reg.refresh()
        _REGISTRY_BY_DIR[model_dir] = reg
    return reg
