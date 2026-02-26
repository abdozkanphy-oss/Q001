from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def load_joblib_model(path: str):
    """Load a sklearn model saved via joblib.

    Kept as a tiny helper to avoid scattering joblib imports and to allow
    caching later if desired.
    """
    import joblib

    return joblib.load(path)


from modules.model_registry import safe_token


_META_RE = re.compile(
    r"^WSUID_(?P<wsuid>.+?)_ST_(?P<st>.+?)_OPTC_(?P<op>.+?)__MIMO_RF__HSEC_(?P<hsec>\d+)__RSEC_(?P<rsec>\d+)__NLAG_(?P<nlag>\d+)__TSET_(?P<tset>[A-Fa-f0-9]+)__meta\.json$"
)


@dataclass(frozen=True)
class MimoRFArtifact:
    wsuid_token: str
    stock_tag: str
    op_tag: str
    horizon_sec: int
    resample_sec: int
    n_lags: int
    tset: str

    model_path: str
    meta_path: str

    # meta
    targets: List[str]
    sensors_used: List[str]
    feature_names: List[str]

    accepted: bool
    lift_mean: Optional[float]
    trained_at_utc: str


class MimoRFRegistry:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._indexed = False
        self._arts: List[MimoRFArtifact] = []

    def refresh(self) -> None:
        self._arts = []
        if not self.model_dir or not os.path.isdir(self.model_dir):
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

            model_path = meta.get("model_path")
            if not model_path:
                model_path = os.path.join(self.model_dir, fn.replace("__meta.json", "__ALG_RF.pkl"))

            metrics = meta.get("metrics") or {}
            accepted = bool(metrics.get("accepted", True))

            def _f(x) -> Optional[float]:
                try:
                    return float(x)
                except Exception:
                    return None

            lift_mean = _f(metrics.get("lift_vs_lag0_mean"))

            trained_at = str(meta.get("trained_at_utc") or "")

            targets = meta.get("targets") or []
            if not isinstance(targets, list):
                targets = []
            sensors_used = meta.get("sensors_used") or []
            if not isinstance(sensors_used, list):
                sensors_used = []
            feature_names = meta.get("feature_names") or []
            if not isinstance(feature_names, list):
                feature_names = []

            art = MimoRFArtifact(
                wsuid_token=str(m.group("wsuid")),
                stock_tag=str(m.group("st")),
                op_tag=str(m.group("op")),
                horizon_sec=int(m.group("hsec")),
                resample_sec=int(m.group("rsec")),
                n_lags=int(m.group("nlag")),
                tset=str(m.group("tset")),
                model_path=str(model_path),
                meta_path=str(meta_path),
                targets=[str(x) for x in targets],
                sensors_used=[str(x) for x in sensors_used],
                feature_names=[str(x) for x in feature_names],
                accepted=bool(accepted),
                lift_mean=lift_mean,
                trained_at_utc=trained_at,
            )
            self._arts.append(art)

        self._indexed = True

    def list_models(self) -> List[MimoRFArtifact]:
        if not self._indexed:
            self.refresh()
        return list(self._arts)

    def find_best(
        self,
        *,
        wsuid: str,
        stock_candidates: List[str],
        op_candidates: List[str],
        resample_sec: Optional[int] = None,
        require_accepted: bool = True,
    ) -> Optional[MimoRFArtifact]:
        if not self._indexed:
            self.refresh()

        wsuid_token = safe_token(wsuid)
        st_tags = [safe_token(s, default="ALL") for s in stock_candidates if str(s).strip()]
        if not st_tags:
            st_tags = ["ALL"]
        op_tags = [safe_token(o, default="ALL") for o in op_candidates if str(o).strip()]
        if not op_tags:
            op_tags = ["ALL"]

        base = [
            a
            for a in self._arts
            if a.wsuid_token == wsuid_token
            and (resample_sec is None or int(a.resample_sec) == int(resample_sec))
        ]
        if not base:
            return None

        if require_accepted:
            acc = [a for a in base if bool(a.accepted)]
            if acc:
                base = acc

        # search ladder: exact stock/op -> stock/ALL -> ALL/op -> ALL/ALL
        ladders: List[Tuple[str, str]] = []
        for st in st_tags:
            for op in op_tags:
                ladders.append((st, op))
            ladders.append((st, "ALL"))
        for op in op_tags:
            ladders.append(("ALL", op))
        ladders.append(("ALL", "ALL"))

        def _score(a: MimoRFArtifact) -> Tuple[int, float, str]:
            # accepted already filtered if required
            lift = float(a.lift_mean) if a.lift_mean is not None else float("-inf")
            trained = a.trained_at_utc or ""
            return (1 if a.accepted else 0, lift, trained)

        for st, op in ladders:
            cands = [a for a in base if a.stock_tag == st and a.op_tag == op]
            if not cands:
                continue
            # best: max accepted/lift/trained_at
            cands = sorted(cands, key=_score, reverse=True)
            return cands[0]

        return None


_REGISTRY_CACHE: Dict[str, MimoRFRegistry] = {}


def get_mimo_rf_registry(model_dir: str) -> MimoRFRegistry:
    model_dir = model_dir or "./models/offline_mimo_rf"
    reg = _REGISTRY_CACHE.get(model_dir)
    if reg is None:
        reg = MimoRFRegistry(model_dir)
        reg.refresh()
        _REGISTRY_CACHE[model_dir] = reg
    return reg
