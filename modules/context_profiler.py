# modules/context_profiler.py
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

_ZERO_STRS = {"0", "0.0", "", "none", "null", "nan"}


def _norm_txt(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.strip().lower() in _ZERO_STRS:
        return None
    return s


def _norm_int(v: Any) -> Optional[str]:
    try:
        iv = int(v)
    except Exception:
        return None
    if iv <= 0:
        return None
    return str(iv)


def _field_values_from_rows(rows: List[Any], field: str) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    is_int = field in {"job_order_operation_id"}
    for r in rows:
        v = getattr(r, field, None)
        out.append(_norm_int(v) if is_int else _norm_txt(v))
    return out


def _stats_for_field(values: List[Optional[str]], *, min_group_size: int = 50, top_k: int = 10) -> Dict[str, Any]:
    n = len(values)
    if n <= 0:
        return {
            "n": 0,
            "coverage": 0.0,
            "distinct": 0,
            "avg_group_size": 0.0,
            "frag_small_group_ratio": 1.0,
            "top": [],
        }

    present = [v for v in values if v is not None]
    cov = float(len(present) / float(n)) if n else 0.0

    if not present:
        return {
            "n": int(n),
            "coverage": cov,
            "distinct": 0,
            "avg_group_size": 0.0,
            "frag_small_group_ratio": 1.0,
            "top": [],
        }

    cnt = Counter(present)
    distinct = int(len(cnt))
    avg_size = float(len(present) / float(distinct)) if distinct else 0.0

    sizes = np.array(list(cnt.values()), dtype="int64")
    frag = float((sizes < int(min_group_size)).mean()) if len(sizes) else 1.0

    top = [{"v": k, "n": int(v)} for k, v in cnt.most_common(int(top_k))]

    return {
        "n": int(n),
        "coverage": float(cov),
        "distinct": int(distinct),
        "avg_group_size": float(avg_size),
        "frag_small_group_ratio": float(frag),
        "top": top,
    }


def _choose_best_variant(candidates: Dict[str, Dict[str, Any]], fields: List[str]) -> Optional[str]:
    """
    Pick the best variant by:
      1) coverage desc
      2) frag_small_group_ratio asc
      3) avg_group_size desc
      4) distinct asc
    """
    best = None
    best_key = None
    for f in fields:
        s = candidates.get(f) or {}
        key = (
            float(s.get("coverage") or 0.0),
            -float(s.get("frag_small_group_ratio") or 1.0),
            float(s.get("avg_group_size") or 0.0),
            -float(s.get("distinct") or 0.0),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = f
    return best


def profile_context_from_rows(
    rows: List[Any],
    *,
    min_group_size: int = 50,
) -> Dict[str, Any]:
    """
    Compute M2.2 context candidate stats from already-fetched dw_tbl_raw_data_by_ws rows.

    Candidates (raw + *_txt variants when present):
      - produced_stock_no
      - prod_order_reference_no, prod_order_reference_no_txt
      - job_order_reference_no, job_order_reference_no_txt
      - job_order_operation_id, job_order_operation_id_txt

    Returns:
      {
        ok: bool,
        candidates: { field_name: stats },
        aliases: { base_field: best_field_variant }
      }
    """
    if not rows:
        return {"ok": False, "reason": "no_rows", "candidates": {}, "aliases": {}}

    fields = [
        "produced_stock_no",
        "prod_order_reference_no",
        "prod_order_reference_no_txt",
        "job_order_reference_no",
        "job_order_reference_no_txt",
        "job_order_operation_id",
        "job_order_operation_id_txt",
    ]

    candidates: Dict[str, Any] = {}
    for field in fields:
        vals = _field_values_from_rows(rows, field)
        candidates[field] = _stats_for_field(vals, min_group_size=min_group_size)

    # Alias groups: choose the best usable variant
    alias_groups = {
        "prod_order_reference_no": ["prod_order_reference_no", "prod_order_reference_no_txt"],
        "job_order_reference_no": ["job_order_reference_no", "job_order_reference_no_txt"],
        "job_order_operation_id": ["job_order_operation_id", "job_order_operation_id_txt"],
    }

    aliases: Dict[str, str] = {}
    for base, variants in alias_groups.items():
        best = _choose_best_variant(candidates, variants)
        if best:
            aliases[base] = best

    return {"ok": True, "candidates": candidates, "aliases": aliases}
