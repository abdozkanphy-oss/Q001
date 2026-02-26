# modules/context_policy.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _ok_candidate(s: Dict[str, Any], *, cov_min: float, distinct_max: int, avg_size_min: float, frag_max: float) -> bool:
    return (
        float(s.get("coverage") or 0.0) >= float(cov_min)
        and int(s.get("distinct") or 0) <= int(distinct_max)
        and float(s.get("avg_group_size") or 0.0) >= float(avg_size_min)
        and (float(s.get("frag_small_group_ratio") or 1.0) <= float(frag_max) or float(s.get("avg_group_size") or 0.0) >= float(avg_size_min) * 10.0)
    )


def _score_candidate(s: Dict[str, Any]) -> Tuple[float, float, float, float]:
    # Higher is better for this tuple
    cov = float(s.get("coverage") or 0.0)
    frag = float(s.get("frag_small_group_ratio") or 1.0)
    avg = float(s.get("avg_group_size") or 0.0)
    distinct = float(s.get("distinct") or 0.0)
    return (cov, -frag, avg, -distinct)


def _pick_best_ok(
    cands: Dict[str, Dict[str, Any]],
    fields: List[str],
    *,
    cov_min: float,
    distinct_max: int,
    avg_size_min: float,
    frag_max: float,
) -> Optional[str]:
    best = None
    best_score = None
    for f in fields:
        s = cands.get(f) or {}
        if not _ok_candidate(s, cov_min=cov_min, distinct_max=distinct_max, avg_size_min=avg_size_min, frag_max=frag_max):
            continue
        sc = _score_candidate(s)
        if best_score is None or sc > best_score:
            best_score = sc
            best = f
    return best


def select_context_policy(
    ctx_stats: Dict[str, Any],
    *,
    prefer_stock: bool = True,
    cov_min: float = 0.20,
    distinct_max: int = 5000,
    avg_size_min: float = 30.0,
    frag_max: float = 0.80,
) -> Dict[str, Any]:
    """
    Returns a policy dict with:
      - context_chain (human-readable)
      - segment_field (used only for series break / segmentation, not for model key)
      - rationale

    Notes:
      - Handles *_txt variants via ctx_stats["aliases"] produced by context_profiler.
      - Fallback is SESSION (gap-based), which is robust under missing IDs.
    """
    cands = (ctx_stats or {}).get("candidates") or {}
    aliases = (ctx_stats or {}).get("aliases") or {}

    # STOCK presence check
    stock = cands.get("produced_stock_no") or {}
    has_stock = float(stock.get("coverage") or 0.0) >= float(cov_min)

    base = ["WS"]
    if prefer_stock and has_stock:
        base.append("STOCK")

    # Candidate preference order (per M2.2):
    # - BATCH: prod_order_reference_no (txt variant if needed)
    # - JOB:   job_order_reference_no (txt variant if needed)
    # - OP:    job_order_operation_id (txt variant if needed)
    order = [
        (["prod_order_reference_no", "prod_order_reference_no_txt"], "BATCH"),
        (["job_order_reference_no", "job_order_reference_no_txt"], "JOB"),
        (["job_order_operation_id", "job_order_operation_id_txt"], "OPERATION"),
    ]

    chosen = None
    chosen_label = None
    tried: List[Dict[str, Any]] = []

    for fields, label in order:
        # prefer alias if provided
        base_field = fields[0]
        alias = aliases.get(base_field)
        pref = [alias] + [f for f in fields if f != alias] if alias else list(fields)

        best = _pick_best_ok(
            cands,
            pref,
            cov_min=cov_min,
            distinct_max=distinct_max,
            avg_size_min=avg_size_min,
            frag_max=frag_max,
        )

        # capture diagnostics for debugging
        diag = {"label": label, "candidates": []}
        for f in pref:
            s = cands.get(f) or {}
            diag["candidates"].append(
                {
                    "field": f,
                    "coverage": float(s.get("coverage") or 0.0),
                    "distinct": int(s.get("distinct") or 0),
                    "avg_group_size": float(s.get("avg_group_size") or 0.0),
                    "frag_small_group_ratio": float(s.get("frag_small_group_ratio") or 1.0),
                }
            )
        tried.append(diag)

        if best is not None:
            chosen = best
            chosen_label = label
            break

    if chosen is None:
        return {
            "segment_field": "SESSION",
            "context_chain": base + ["SESSION"],
            "rationale": {
                "reason": "no_reliable_ids",
                "thresholds": {
                    "cov_min": cov_min,
                    "distinct_max": distinct_max,
                    "avg_size_min": avg_size_min,
                    "frag_max": frag_max,
                },
                "tried": tried,
                "aliases": aliases,
            },
        }

    return {
        "segment_field": chosen,
        "context_chain": base + [chosen_label],
        "rationale": {
            "reason": f"{chosen_label.lower()}_meets_thresholds",
            "thresholds": {
                "cov_min": cov_min,
                "distinct_max": distinct_max,
                "avg_size_min": avg_size_min,
                "frag_max": frag_max,
            },
            "tried": tried,
            "aliases": aliases,
        },
    }
