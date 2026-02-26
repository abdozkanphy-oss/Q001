# modules/model_leaderboard.py
from __future__ import annotations

import argparse

from modules.model_registry import get_outonly_registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./models/offline_outonly")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--min_test", type=int, default=20)
    ap.add_argument("--accepted_only", action="store_true", help="Show only acceptance-passing models")
    args = ap.parse_args()

    reg = get_outonly_registry(args.dir)
    arts = reg.list_outonly()

    if args.accepted_only:
        arts = [a for a in arts if bool(getattr(a, "accepted", True))]

    # sort by mae (None last), then by n_test desc
    def key(a):
        mae = a.mae if a.mae is not None else 1e18
        return (mae, -int(a.n_test or 0))

    arts = sorted(arts, key=key)[: int(args.top)]

    print(f"FOUND outonly_models={len(reg.list_outonly())} shown={len(arts)} accepted_only={bool(args.accepted_only)}")
    print("TOP MODELS (lowest MAE):")
    for a in arts:
        print(
            f"mae={a.mae} base_mae={a.baseline_mae_lag0} lift={a.lift_vs_lag0} "
            f"acc={a.accepted} eval={a.eval_mode} n_test={a.n_test} n_train={a.n_train} "
            f"FM={a.feature_mode} "
            f"WSUID={a.wsuid_token} ST={a.stock_tag} OPTC={a.op_tag} "
            f"TGT={a.target_tag} HSEC={a.horizon_sec} RSEC={a.resample_sec} "
            f"trained_at={a.trained_at_utc}"
        )


if __name__ == "__main__":
    main()
