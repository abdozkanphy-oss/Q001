import os
import re
import json
import argparse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

from utils.config_reader import ConfigReader
from cassandra_utils.models.dw_raw_data import dw_raw_data

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_token(x, default="UNKNOWN"):
    if x is None:
        return default
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return default
    return _SAFE_TOKEN_RE.sub("_", s)


def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def parse_inputs(inputvariablelist):
    feats = {}
    for d in inputvariablelist or []:
        if not isinstance(d, dict):
            continue
        name = d.get("varNm") or d.get("varId")
        if not name:
            continue
        val = d.get("genReadVal")
        if val in (None, "", "None"):
            val = d.get("actVal")
        v = to_float(val)
        if v is None:
            continue

        var_cat = d.get("varCat")
        if var_cat:
            col = f"in_{safe_token(var_cat)}__{safe_token(name)}"
        else:
            col = f"in_{safe_token(name)}"
        feats[col] = v
    return feats


def parse_outputs(outputvaluelist):
    outs = {}
    for d in outputvaluelist or []:
        if not isinstance(d, dict):
            continue
        eq_nm = d.get("eqNm") or d.get("eqNo")
        eq_id = d.get("eqId")
        if not eq_nm and not eq_id:
            continue

        val = d.get("cntRead")
        if val in (None, "", "None"):
            val = d.get("genReadVal") or d.get("actVal")
        v = to_float(val)
        if v is None:
            continue

        if eq_nm and eq_id is not None:
            col = f"out_{safe_token(eq_nm)}__id{safe_token(eq_id)}"
        elif eq_nm:
            col = f"out_{safe_token(eq_nm)}"
        else:
            col = f"out_id{safe_token(eq_id)}"
        outs[col] = v
    return outs


def fetch_rows(pl_id, wc_id, ws_id, start_dt, end_dt, op_tc=None, st_id=None, limit=50000):
    q = dw_raw_data.objects(partition_key="latest").allow_filtering()

    # Core workstation filters
    q = q.filter(plantid=int(pl_id), workcenterid=int(wc_id), workstationid=int(ws_id))

    # Optional filters
    if op_tc is not None and str(op_tc).strip() != "":
        q = q.filter(operationtaskcode=str(op_tc))
    if st_id is not None and str(st_id).strip() != "":
        q = q.filter(outputstockid=int(st_id))

    # Time filters
    if start_dt is not None:
        q = q.filter(measurement_date__gte=start_dt)
    if end_dt is not None:
        q = q.filter(measurement_date__lte=end_dt)

    # Cassandra clustering is DESC; we’ll sort later.
    rows = list(q.limit(int(limit)))
    return rows


def build_frame(rows):
    recs = []
    for r in rows:
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue
        feats = parse_inputs(getattr(r, "inputvariablelist", None))
        outs  = parse_outputs(getattr(r, "outputvaluelist", None))
        if not feats and not outs:
            continue
        recs.append({"ts": ts, **feats, **outs})

    if not recs:
        return pd.DataFrame()

    df = pd.DataFrame(recs).set_index("ts").sort_index()
    # drop duplicate timestamps by keeping last
    df = df[~df.index.duplicated(keep="last")]
    return df


def train_offline_exo(df, resample_sec, horizon_steps, horizon_sec,
                      min_rows=500, max_missing_feat=0.5, max_missing_out=0.5,
                      feat_ffill_limit=1, out_ffill_limit=1):
    if df.empty:
        return None

    freq = f"{int(resample_sec)}S"
    df_r = df.resample(freq).last()

    feat_cols = [c for c in df_r.columns if c.startswith("in_")]
    out_cols  = [c for c in df_r.columns if c.startswith("out_")]

    if not feat_cols or not out_cols:
        return None

    X = df_r[feat_cols].apply(pd.to_numeric, errors="coerce")
    Y = df_r[out_cols].apply(pd.to_numeric, errors="coerce")

    # mild fills for inputs/outputs to improve alignment without long leakage
    if feat_ffill_limit and feat_ffill_limit > 0:
        X = X.ffill(limit=int(feat_ffill_limit))
    if out_ffill_limit and out_ffill_limit > 0:
        Y = Y.ffill(limit=int(out_ffill_limit))

    # drop sparse columns
    feat_keep = (X.isna().mean() <= float(max_missing_feat))
    out_keep  = (Y.isna().mean() <= float(max_missing_out))

    X = X.loc[:, feat_keep[feat_keep].index]
    Y = Y.loc[:, out_keep[out_keep].index]

    # shift labels to represent horizon
    Yh = Y.shift(-int(horizon_steps))

    # remove tail rows without labels
    XY = pd.concat([X, Yh], axis=1)
    XY = XY.dropna(axis=0, how="any")

    if len(XY) < int(min_rows):
        return None

    X2 = XY[X.columns]
    Y2 = XY[Yh.columns]

    # time split
    n = len(X2)
    split = int(n * 0.8)
    X_tr, X_te = X2.iloc[:split], X2.iloc[split:]
    Y_tr, Y_te = Y2.iloc[:split], Y2.iloc[split:]

    base = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_tr, Y_tr)

    Y_hat = model.predict(X_te)
    metrics = {}
    for j, col in enumerate(Y2.columns):
        mae = float(mean_absolute_error(Y_te.iloc[:, j].values, Y_hat[:, j]))
        metrics[col] = {"mae": mae}

    return {
        "model": model,
        "feature_cols": list(X2.columns),
        "target_cols": list(Y2.columns),
        "metrics": metrics,
        "n_rows": int(n),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "resample_sec": int(resample_sec),
        "horizon_steps": int(horizon_steps),
        "horizon_sec": int(horizon_sec),
    }


def main():
    cfg = ConfigReader()
    resample_sec = int(getattr(cfg, "resample_seconds", 60) or 60)
    horizon_steps = int(getattr(cfg, "prediction_horizon_steps", 1) or 1)
    horizon_sec = int(getattr(cfg, "prediction_horizon_sec", 0) or resample_sec)

    ap = argparse.ArgumentParser()
    ap.add_argument("--plId", required=True, type=int)
    ap.add_argument("--wcId", required=True, type=int)
    ap.add_argument("--wsId", required=True, type=int)
    ap.add_argument("--opTc", default=None)
    ap.add_argument("--stId", default=None)
    ap.add_argument("--days", default=7, type=int)
    ap.add_argument("--limit", default=50000, type=int)
    ap.add_argument("--out_dir", default="./models/offline_exo")
    args = ap.parse_args()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(args.days))

    rows = fetch_rows(args.plId, args.wcId, args.wsId, start_dt, end_dt,
                      op_tc=args.opTc, st_id=args.stId, limit=args.limit)
    df = build_frame(rows)

    res = train_offline_exo(df, resample_sec, horizon_steps, horizon_sec)
    if res is None:
        print("TRAIN_SKIPPED: insufficient data after alignment/filters.")
        return

    ws_uid = f"{args.plId}_WC{args.wcId}_WS{args.wsId}"
    key = f"WSUID_{safe_token(ws_uid)}"
    if args.opTc:
        key += f"_OPTC_{safe_token(args.opTc)}"
    if args.stId:
        key += f"_STID_{safe_token(args.stId)}"
    key += f"__EXO_OUT__HSEC_{int(horizon_sec)}"

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, f"{key}__ALG_RF.pkl")
    meta_path  = os.path.join(args.out_dir, f"{key}__meta.json")

    joblib.dump(res["model"], model_path)
    meta = {k: v for k, v in res.items() if k != "model"}
    meta["model_path"] = model_path
    meta["trained_at_utc"] = datetime.now(timezone.utc).isoformat()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"TRAIN_OK: saved model={model_path}")
    print(f"TRAIN_OK: saved meta ={meta_path}")


if __name__ == "__main__":
    main()