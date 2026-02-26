# modules/import_csv_to_dw_raw_by_ws.py
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from cassandra.cqlengine.query import BatchQuery

from cassandra_utils.models.dw_raw_by_ws import dw_tbl_raw_data_by_ws, ensure_dw_raw_by_ws

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


REQUIRED_COLS = [
    "plant_id",
    "work_center_id",
    "work_station_id",
    "measurement_date",
    "unique_code",
]


# ---------- type coercions ----------

def _is_nan(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return x is None


def _to_str(x: Any) -> Optional[str]:
    if _is_nan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return None
    return s


def _to_int(x: Any) -> Optional[int]:
    if _is_nan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return None
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def _to_float(x: Any) -> Optional[float]:
    if _is_nan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return None
    try:
        v = float(s)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _to_bool(x: Any) -> Optional[bool]:
    if _is_nan(x):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "t", "yes", "y"):
        return True
    if s in ("false", "0", "f", "no", "n"):
        return False
    return None


def _parse_dt(x: Any) -> Optional[datetime]:
    """
    Accepts:
      - '2026-01-20 13:26:36.207000'
      - ISO strings with timezone
    Returns tz-naive datetime for Cassandra.
    """
    if _is_nan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return None
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=False)
        if pd.isna(ts):
            return None
        dt = ts.to_pydatetime()
        # Normalize to tz-naive for Cassandra DateTime
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        return None


# ---------- stats / checkpoint ----------

@dataclass
class ImportStats:
    rows_seen: int = 0
    rows_ok: int = 0
    rows_skipped: int = 0
    rows_failed: int = 0

    missing_required: int = 0
    bad_datetime: int = 0
    bad_key: int = 0
    mismatch_filter: int = 0


def _load_checkpoint(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_checkpoint(path: Optional[str], data: Dict[str, Any]) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _iter_csv_files(csv_path: Optional[str], csv_glob: Optional[str]) -> List[str]:
    files: List[str] = []
    if csv_path:
        files.append(csv_path)
    if csv_glob:
        files.extend(sorted(glob.glob(csv_glob)))
    files = [f for f in files if f and os.path.exists(f)]
    out = []
    seen = set()
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def _validate_required_cols(csv_cols: List[str]) -> Tuple[bool, List[str]]:
    s = set(csv_cols)
    missing = [c for c in REQUIRED_COLS if c not in s]
    return (len(missing) == 0, missing)


def _model_columns() -> Dict[str, Any]:
    """
    Returns model column map {name: Column}.
    cqlengine stores columns in Model._columns.
    """
    return dict(dw_tbl_raw_data_by_ws._columns)


# ---------- row mapping ----------
# We map known columns with explicit coercion based on expected semantics.
# For any unknown-but-present columns: they will still be preserved in raw_row_json (if enabled).

# dw_tbl_raw_data_by_ws (NEW schema) type sets

INT_COLS = {
    # partition keys (int)
    "plant_id",
    "work_center_id",
    "work_station_id",

    # numeric ids (int)
    "equipment_id",
    "produced_stock_id",
    "job_order_operation_id",
    "stock_id",
}

# These are doubles in schema
FLOAT_COLS = {
    "counter_reading",
    "gen_read_val",
}

BOOL_COLS = {
    "equipment_type",
    "active",
    "good",
    "anomaly_detection_active",
    "trend_calculation_active",
}

DT_COLS = {
    "measurement_date",   # clustering timestamp
    "create_date",
    "update_date",
    "shift_start_time",
    "shift_finish_time",
}

# Everything that is text in your NEW by_ws table, including "fixed" columns
TEXT_COLS = {
    # keys / identifiers
    "unique_code",
    "customer",

    # location / org
    "plant_name",
    "work_center_name",
    "work_center_no",
    "work_station_name",
    "work_station_no",

    # states / codes
    "machine_state",
    "workstation_state",
    "valuation_code",

    # production
    "produced_stock_no",
    "produced_stock_name",
    "operationname",
    "operationno",
    "operationtaskcode",

    # equipment
    "equipment_name",
    "equipment_no",

    # raw snapshot
    "raw_row_json",

    # FIXED TYPES (were int before; now text)
    "prod_order_reference_no",
    "job_order_reference_no",
    "year",
}



def _coerce_value(col: str, val: Any) -> Any:
    if col in DT_COLS:
        return _parse_dt(val)
    if col in INT_COLS:
        return _to_int(val)
    if col in FLOAT_COLS:
        return _to_float(val)
    if col in BOOL_COLS:
        return _to_bool(val)
    # default to string for everything else
    return _to_str(val)


def _row_to_record(
    row: Dict[str, Any],
    model_cols: Dict[str, Any],
    enforce_pl: Optional[int],
    enforce_wc: Optional[int],
    enforce_ws: Optional[int],
    include_raw_json: bool,
    stats: ImportStats,
) -> Optional[Dict[str, Any]]:
    # required keys
    plant_id = _to_int(row.get("plant_id"))
    wc_id = _to_int(row.get("work_center_id"))
    ws_id = _to_int(row.get("work_station_id"))
    meas_dt = _parse_dt(row.get("measurement_date"))
    unique_code = _to_str(row.get("unique_code"))

    if plant_id is None or wc_id is None or ws_id is None or meas_dt is None or not unique_code:
        stats.missing_required += 1
        if meas_dt is None:
            stats.bad_datetime += 1
        if not unique_code:
            stats.bad_key += 1
        return None

    # enforcement filters
    if enforce_pl is not None and plant_id != int(enforce_pl):
        stats.mismatch_filter += 1
        return None
    if enforce_wc is not None and wc_id != int(enforce_wc):
        stats.mismatch_filter += 1
        return None
    if enforce_ws is not None and ws_id != int(enforce_ws):
        stats.mismatch_filter += 1
        return None

    rec: Dict[str, Any] = {
        "plant_id": int(plant_id),
        "work_center_id": int(wc_id),
        "work_station_id": int(ws_id),
        "measurement_date": meas_dt,
        "unique_code": str(unique_code),
    }

    # map all CSV columns that exist in the model (except required which are already set)
    for col in row.keys():
        if col in REQUIRED_COLS:
            continue
        if col not in model_cols:
            continue
        if col == "raw_row_json":
            # handled separately below
            continue
        rec[col] = _coerce_value(col, row.get(col))

    # Optional: preserve full row JSON if the table has raw_row_json
    if include_raw_json and "raw_row_json" in model_cols:
        try:
            rec["raw_row_json"] = json.dumps(row, ensure_ascii=False, default=str)
        except Exception:
            rec["raw_row_json"] = None

    return rec


def _batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------- import core ----------

def import_one_csv(
    csv_file: str,
    chunksize: int,
    batch_size: int,
    checkpoint: Optional[str],
    dry_run: bool,
    enforce_pl: Optional[int],
    enforce_wc: Optional[int],
    enforce_ws: Optional[int],
    max_rows: int,
    sleep_ms_between_batches: int,
    include_raw_json: bool,
    skip_done_files: bool,
) -> ImportStats:
    stats = ImportStats()

    ck = _load_checkpoint(checkpoint)
    done_files = set(ck.get("done_files") or [])
    if skip_done_files and csv_file in done_files:
        print(f"SKIP (checkpoint done): {csv_file}")
        return stats

    # validate header
    header_df = pd.read_csv(csv_file, nrows=0)
    csv_cols = header_df.columns.tolist()
    ok, missing = _validate_required_cols(csv_cols)
    if not ok:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    # Determine columns to read: intersection(CSV, Model)
    model_cols = _model_columns()
    usecols = sorted(list(set(csv_cols).intersection(set(model_cols.keys()))))

    # Ensure required are available in CSV (already checked) AND in model
    # If required keys aren't in the model, inserts cannot work.
    for c in REQUIRED_COLS:
        if c not in model_cols:
            raise RuntimeError(
                f"Model dw_tbl_raw_data_by_ws is missing required key column '{c}'. "
                f"Fix the model/table schema first."
            )

    # Read in chunks; keep strings to avoid pandas casting weirdness
    reader = pd.read_csv(
        csv_file,
        chunksize=int(chunksize),
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_values=["", "null", "NULL", "None", "none", "NaN", "nan"],
    )

    iterator = reader
    if tqdm is not None:
        iterator = tqdm(reader, desc=f"Import {os.path.basename(csv_file)}", unit="chunk")


    # Do NOT run schema management during imports.
    # It can be slow and it will warn/block when existing Cassandra types differ.
    # Run schema sync explicitly (separately) when you intend to evolve the schema.
    if os.getenv("CQLENG_ALLOW_SCHEMA_MANAGEMENT") == "1":
        try:
            ensure_dw_raw_by_ws()
        except Exception as e:
            print(f"WARN: ensure_dw_raw_by_ws failed: {e}")
    else:
        # Table must already exist; importer proceeds with inserts.
        pass


    total_seen = 0

    for chunk in iterator:
        if max_rows and total_seen >= int(max_rows):
            break

        # normalize nan->None
        chunk = chunk.replace({np.nan: None})
        rows = chunk.to_dict(orient="records")
        total_seen += len(rows)

        recs: List[Dict[str, Any]] = []
        for r in rows:
            stats.rows_seen += 1
            rec = _row_to_record(
                row=r,
                model_cols=model_cols,
                enforce_pl=enforce_pl,
                enforce_wc=enforce_wc,
                enforce_ws=enforce_ws,
                include_raw_json=include_raw_json,
                stats=stats,
            )
            if rec is None:
                stats.rows_skipped += 1
                continue
            recs.append(rec)

        if dry_run:
            stats.rows_ok += len(recs)
            continue

        for b in _batched(recs, int(batch_size)):
            # Batch insert (upsert)
            ok_in_batch = 0
            try:
                with BatchQuery() as batch:
                    for rec in b:
                        # create = upsert by PK in Cassandra
                        dw_tbl_raw_data_by_ws.batch(batch).create(**rec)
                        ok_in_batch += 1
                stats.rows_ok += ok_in_batch
            except Exception:
                # fallback row-by-row
                for rec in b:
                    try:
                        dw_tbl_raw_data_by_ws.create(**rec)
                        stats.rows_ok += 1
                    except Exception:
                        stats.rows_failed += 1

            if sleep_ms_between_batches > 0:
                time.sleep(float(sleep_ms_between_batches) / 1000.0)

    # checkpoint update
    if checkpoint:
        ck = _load_checkpoint(checkpoint)
        ck.setdefault("done_files", [])
        if csv_file not in ck["done_files"]:
            ck["done_files"].append(csv_file)
        ck["last_run_utc"] = datetime.utcnow().isoformat()
        ck["skip_done_files"] = bool(skip_done_files)
        _save_checkpoint(checkpoint, ck)

    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Import dw_tbl_raw_data CSV exports into dw_tbl_raw_data_by_ws (workstation/time projection)."
    )

    ap.add_argument("--csv", default=None, help="Single CSV file path")
    ap.add_argument("--glob", dest="csv_glob", default=None, help="Glob for multiple CSV files")

    ap.add_argument("--chunksize", type=int, default=50000, help="CSV read chunksize (rows)")
    ap.add_argument("--batch_size", type=int, default=500, help="Cassandra batch size (rows)")

    ap.add_argument("--checkpoint", default="./import_dw_by_ws_checkpoint.json", help="Checkpoint JSON path")
    ap.add_argument("--skip_done_files", action="store_true", help="Skip files already listed in checkpoint")
    ap.add_argument("--no_raw_json", action="store_true", help="Do not store raw_row_json even if table has it")

    ap.add_argument("--dry_run", action="store_true", help="Parse/validate only; no DB writes")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional hard cap rows to process (0=all)")
    ap.add_argument("--sleep_ms", type=int, default=0, help="Sleep between Cassandra batches (ms)")

    # Optional enforcement: drop rows not matching these IDs
    ap.add_argument("--plId", type=int, default=None)
    ap.add_argument("--wcId", type=int, default=None)
    ap.add_argument("--wsId", type=int, default=None)

    args = ap.parse_args()

    files = _iter_csv_files(args.csv, args.csv_glob)
    if not files:
        print("ERROR: no CSV files found. Use --csv or --glob.")
        sys.exit(2)

    include_raw_json = not bool(args.no_raw_json)

    print(f"FILES={len(files)} dry_run={args.dry_run} chunksize={args.chunksize} batch_size={args.batch_size}")
    print(f"ENFORCE plId={args.plId} wcId={args.wcId} wsId={args.wsId}")
    print(f"CHECKPOINT={args.checkpoint} skip_done_files={args.skip_done_files} include_raw_json={include_raw_json}")
    print("NOTE: If you see schema-management warnings, you can set CQLENG_ALLOW_SCHEMA_MANAGEMENT=1 for sync_table().")

    grand = ImportStats()

    for f in files:
        print(f"\n=== IMPORT START: {f} ===")
        s = import_one_csv(
            csv_file=f,
            chunksize=int(args.chunksize),
            batch_size=int(args.batch_size),
            checkpoint=args.checkpoint,
            dry_run=bool(args.dry_run),
            enforce_pl=args.plId,
            enforce_wc=args.wcId,
            enforce_ws=args.wsId,
            max_rows=int(args.max_rows),
            sleep_ms_between_batches=int(args.sleep_ms),
            include_raw_json=include_raw_json,
            skip_done_files=bool(args.skip_done_files),
        )

        print(
            f"=== IMPORT END: {os.path.basename(f)} "
            f"seen={s.rows_seen} ok={s.rows_ok} skipped={s.rows_skipped} failed={s.rows_failed} "
            f"missing_required={s.missing_required} bad_dt={s.bad_datetime} bad_key={s.bad_key} mismatch_filter={s.mismatch_filter}"
        )

        # accumulate
        for k in grand.__dict__.keys():
            setattr(grand, k, getattr(grand, k) + getattr(s, k))

    print(
        f"\nTOTAL seen={grand.rows_seen} ok={grand.rows_ok} skipped={grand.rows_skipped} failed={grand.rows_failed} "
        f"missing_required={grand.missing_required} bad_dt={grand.bad_datetime} bad_key={grand.bad_key} mismatch_filter={grand.mismatch_filter}"
    )


if __name__ == "__main__":
    main()