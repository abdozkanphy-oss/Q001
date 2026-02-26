# modules/offline_outonly_sweep.py
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from modules.workstation_profile import build_workstation_profile


def _run(cmd: List[str]) -> int:
    print("CMD:", " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except Exception as e:
        print("ERROR running command:", e)
        return 2


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _best_result(per_target: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    acc = [x for x in per_target if bool(x.get("accepted")) and x.get("model_mae") is not None]
    if acc:
        return sorted(acc, key=lambda x: float(x["model_mae"]))[0]
    any_mae = [x for x in per_target if x.get("model_mae") is not None]
    if any_mae:
        return sorted(any_mae, key=lambda x: float(x["model_mae"]))[0]
    return per_target[0] if per_target else None


def main():
    ap = argparse.ArgumentParser(description="Offline OUT_ONLY sweep with Option-2 fallback (UNI -> MV if needed).")
    ap.add_argument("--ws_list", default="", help="path to ws_list.txt (lines: plId,wcId,wsId)")
    ap.add_argument("--out_dir", default="./models/offline_outonly")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--resample_sec", type=int, default=60)
    ap.add_argument("--horizon_sec", type=int, default=60)
    ap.add_argument("--n_lags", type=int, default=6)
    ap.add_argument("--min_rows", type=int, default=80)
    ap.add_argument("--top_targets", type=int, default=5)

    # Eval / acceptance (forwarded to trainer)
    ap.add_argument("--eval_mode", default="auto", choices=["auto", "time_split", "batch_holdout"])
    ap.add_argument("--holdout_k", type=int, default=3)
    ap.add_argument("--min_points_per_batch", type=int, default=50)
    ap.add_argument("--min_test", type=int, default=20)
    ap.add_argument("--accept_min_lift", type=float, default=0.0)

    # Option-2 controls
    ap.add_argument("--try_multivariate", action="store_true", help="If UNI not accepted, try MV for those targets")
    ap.add_argument("--max_sensors", type=int, default=50)
    ap.add_argument("--min_sensor_non_nan", type=int, default=10)

    # Context controls
    ap.add_argument("--stNo", default="ALL")
    ap.add_argument("--opTc", default="ALL")

    ap.add_argument("--report", default="outonly_sweep_report.json")

    args = ap.parse_args()

    out_dir = str(args.out_dir or "./models/offline_outonly")
    os.makedirs(out_dir, exist_ok=True)

    # Parse ws list
    ws_items: List[Tuple[int, int, int]] = []
    if args.ws_list:
        with open(args.ws_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    ws_items.append((int(parts[0]), int(parts[1]), int(parts[2])))
                except Exception:
                    continue

    if not ws_items:
        print("No workstations in ws_list.")
        return 2

    sweep_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report: Dict[str, Any] = {
        "sweep_id": sweep_id,
        "ts_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "args": vars(args),
        "plans": [],
        "acceptance_summary": {},
    }

    total_models = 0
    total_accepted = 0
    total_skipped_baseline_perfect = 0
    total_skipped_other = 0
    best_by_target: Dict[str, Dict[str, Any]] = {}

    for pl_id, wc_id, ws_id in ws_items:
        # profile to pick targets
        prof = build_workstation_profile(
            pl_id, wc_id, ws_id,
            days=int(args.days),
            limit=int(args.limit),
            st_no=None if str(args.stNo).upper() == "ALL" else str(args.stNo),
            op_tc=None if str(args.opTc).upper() == "ALL" else str(args.opTc),
            resample_sec=int(args.resample_sec),
        )
        if not prof.get("ok"):
            report["plans"].append({"ws": [pl_id, wc_id, ws_id], "ok": False, "profile": prof})
            continue

        targets = list(prof.get("out_targets_top") or [])[: int(args.top_targets)]
        if not targets:
            # Fallback: use top sensors by raw count when coverage-based candidates are empty (bursty / batch-driven customers)
            sensors_top = [x.get("sensor") for x in (prof.get("sensors_top") or []) if x.get("sensor")]
            targets = sensors_top[: int(args.top_targets)]
        if not targets:
            report["plans"].append({"ws": [pl_id, wc_id, ws_id], "ok": False, "profile": prof, "reason": "no_targets"})
            continue

        # UNI run
        run_id_uni = f"{sweep_id}_UNI_{pl_id}_{wc_id}_{ws_id}"
        cmd_uni = [
            "python", "-m", "modules.offline_outonly_trainer",
            "--plId", str(pl_id),
            "--wcId", str(wc_id),
            "--wsId", str(ws_id),
            "--days", str(int(args.days)),
            "--limit", str(int(args.limit)),
            "--stNo", str(args.stNo),
            "--opTc", str(args.opTc),
            "--targets", ",".join(targets),
            "--min_rows", str(int(args.min_rows)),
            "--n_lags", str(int(args.n_lags)),
            "--resample_sec", str(int(args.resample_sec)),
            "--horizon_sec", str(int(args.horizon_sec)),
            "--feature_mode", "univariate",
            "--eval_mode", str(args.eval_mode),
            "--holdout_k", str(int(args.holdout_k)),
            "--min_points_per_batch", str(int(args.min_points_per_batch)),
            "--min_test", str(int(args.min_test)),
            "--accept_min_lift", str(float(args.accept_min_lift)),
            "--out_dir", out_dir,
            "--run_id", run_id_uni,
        ]
        rc_uni = _run(cmd_uni)

        # trainer writes deterministic run summary path
        # We don't know wsuid_token here without duplicating logic; just scan for newest summary with run_id_uni suffix
        uni_summary_path = None
        for fn in os.listdir(out_dir):
            if fn.endswith(f"{run_id_uni}.json") and fn.startswith("run_outonly_"):
                uni_summary_path = os.path.join(out_dir, fn)
                break
        uni_summary = _read_json(uni_summary_path) if uni_summary_path else None

        per_target_results: List[Dict[str, Any]] = []
        if uni_summary and isinstance(uni_summary.get("targets"), list):
            per_target_results.extend([t for t in uni_summary["targets"] if t.get("ok")])

        # Determine MV fallback set
        mv_summary_path = None
        mv_summary = None
        if bool(args.try_multivariate) and uni_summary:
            need_mv = [t["target"] for t in per_target_results if (not bool(t.get("accepted"))) and (not bool(t.get("skipped"))) ]
            if need_mv:
                run_id_mv = f"{sweep_id}_MV_{pl_id}_{wc_id}_{ws_id}"
                cmd_mv = [
                    "python", "-m", "modules.offline_outonly_trainer",
                    "--plId", str(pl_id),
                    "--wcId", str(wc_id),
                    "--wsId", str(ws_id),
                    "--days", str(int(args.days)),
                    "--limit", str(int(args.limit)),
                    "--stNo", str(args.stNo),
                    "--opTc", str(args.opTc),
                    "--targets", ",".join(need_mv),
                    "--min_rows", str(int(args.min_rows)),
                    "--n_lags", str(int(args.n_lags)),
                    "--resample_sec", str(int(args.resample_sec)),
                    "--horizon_sec", str(int(args.horizon_sec)),
                    "--feature_mode", "multivariate",
                    "--max_sensors", str(int(args.max_sensors)),
                    "--min_sensor_non_nan", str(int(args.min_sensor_non_nan)),
                    "--eval_mode", str(args.eval_mode),
                    "--holdout_k", str(int(args.holdout_k)),
                    "--min_points_per_batch", str(int(args.min_points_per_batch)),
                    "--min_test", str(int(args.min_test)),
                    "--accept_min_lift", str(float(args.accept_min_lift)),
                    "--out_dir", out_dir,
                    "--run_id", run_id_mv,
                ]
                rc_mv = _run(cmd_mv)
                for fn in os.listdir(out_dir):
                    if fn.endswith(f"{run_id_mv}.json") and fn.startswith("run_outonly_"):
                        mv_summary_path = os.path.join(out_dir, fn)
                        break
                mv_summary = _read_json(mv_summary_path) if mv_summary_path else None
                if mv_summary and isinstance(mv_summary.get("targets"), list):
                    per_target_results.extend([t for t in mv_summary["targets"] if t.get("ok")])

        # acceptance aggregation for this ws plan
        tried = [t for t in per_target_results if t.get("model_mae") is not None]
        accepted = [t for t in tried if bool(t.get("accepted"))]
        total_models += len(tried)
        total_accepted += len(accepted)
        skipped_bp = [t for t in per_target_results if bool(t.get("skipped")) and str(t.get("skip_reason") or "") == "baseline_perfect"]
        skipped_other = [t for t in per_target_results if bool(t.get("skipped")) and str(t.get("skip_reason") or "") != "baseline_perfect"]
        total_skipped_baseline_perfect += len(skipped_bp)
        total_skipped_other += len(skipped_other)

        # best per target (within this sweep plan)
        # key by target string only (since st/op fixed per sweep args)
        by_tgt: Dict[str, List[Dict[str, Any]]] = {}
        for t in per_target_results:
            by_tgt.setdefault(str(t.get("target")), []).append(t)

        best_local = []
        for tgt, lst in by_tgt.items():
            b = _best_result(lst)
            if b is None:
                continue
            best_local.append(b)

            # global key: (pl,wc,ws,st,op,target,h,r)
            gk = f"{pl_id}:{wc_id}:{ws_id}|{args.stNo}|{args.opTc}|{tgt}|H{args.horizon_sec}|R{args.resample_sec}"
            cand = {
                "ws": [pl_id, wc_id, ws_id],
                "stNo": str(args.stNo),
                "opTc": str(args.opTc),
                "target": tgt,
                "best_feature_mode": "SKIP" if bool(b.get("skipped")) else ("MV" if "FM_MV" in str(b.get("meta_path") or "") else "UNI"),
                "accepted": bool(b.get("accepted")),
                "skipped": bool(b.get("skipped")),
                "skip_reason": b.get("skip_reason"),
                "reject_reason": b.get("reject_reason"),
                "model_mae": b.get("model_mae"),
                "baseline_mae_lag0": b.get("baseline_mae_lag0"),
                "lift_vs_lag0": b.get("lift_vs_lag0"),
                "eval_mode": b.get("eval_mode"),
                "n_test": int(b.get("n_test") or 0),
                "meta_path": b.get("meta_path"),
                "model_path": b.get("model_path"),
            }
            prev = best_by_target.get(gk)
            if prev is None:
                best_by_target[gk] = cand
            else:
                # prefer accepted, then lower mae
                if (not prev.get("accepted")) and cand.get("accepted"):
                    best_by_target[gk] = cand
                elif bool(prev.get("accepted")) == bool(cand.get("accepted")):
                    try:
                        if cand.get("model_mae") is not None and prev.get("model_mae") is not None and float(cand["model_mae"]) < float(prev["model_mae"]):
                            best_by_target[gk] = cand
                    except Exception:
                        pass

        plan = {
            "ws": [pl_id, wc_id, ws_id],
            "ok": True,
            "profile": prof,
            "targets_requested": targets,
            "uni": {"rc": int(rc_uni), "run_id": run_id_uni, "summary_path": uni_summary_path},
            "mv": {"run_id": None, "summary_path": mv_summary_path},
            "results": per_target_results,
            "best_per_target": best_local,
        }
        report["plans"].append(plan)

    # lightweight acceptance summary
    best_list = list(best_by_target.values())
    best_list_sorted = sorted(
        best_list,
        key=lambda x: (
            0 if bool(x.get("accepted")) else 1,
            float(x.get("model_mae") or 1e18),
        ),
    )

    report["acceptance_summary"] = {
        "models_total_evaluated": int(total_models),
        "models_accepted": int(total_accepted),
        "accepted_rate": float(total_accepted / total_models) if total_models else 0.0,
        "models_skipped_baseline_perfect": int(total_skipped_baseline_perfect),
        "models_skipped_other": int(total_skipped_other),
        "best_by_target": best_list_sorted,
    }

    report_path = str(args.report)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"SWEEP REPORT written: {report_path}")
    print(f"ACCEPTANCE: total={total_models} accepted={total_accepted} rate={report['acceptance_summary']['accepted_rate']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
