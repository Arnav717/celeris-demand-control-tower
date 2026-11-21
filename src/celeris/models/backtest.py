"""Rolling-origin backtest with the full 19-month horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd

from celeris.features.build import MIDX, MONTHS, build_frame
from celeris.models.baselines import BASELINES
from celeris.models.gbm import GBM

CUTOFFS = ["2023-10", "2024-04", "2024-10", "2025-03"]  # last origin with 19 months of actuals


def wape(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.abs(y - f).sum() / max(y.sum(), 1e-9))


def bias(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float((f - y).sum() / max(y.sum(), 1e-9))


GBM_SPECS = {
    "gbm_tweedie": {"objective": "tweedie", "params": {"tweedie_variance_power": 1.3}, "rounds": 600},
    "gbm_l1": {"objective": "l1", "rounds": 600},
}


def run_backtest(P, X, train_frame: pd.DataFrame, cutoffs=CUTOFFS, quantile_cutoff: str | None = "2024-10"):
    preds = []
    for c in cutoffs:
        t = MIDX[c]
        va = build_frame(P, X, [t])
        va = va[va["y_demand"].notna()].copy()
        tr = train_frame[train_frame["target_m"] <= t]
        out = va[["series", "origin", "target_m", "h", "y_demand"]].copy()
        for name, fn in BASELINES.items():
            F = fn(P.Y, t)
            out[name] = F[va["series"], va["h"].astype(int) - 1]
        for name, spec in GBM_SPECS.items():
            out[name] = GBM(**spec).fit(tr).predict(va)
        out["ensemble"] = 0.5 * (out["gbm_tweedie"] + out["gbm_l1"])
        if c == quantile_cutoff:
            for q in (0.1, 0.9):
                out[f"q{int(q * 100)}"] = GBM("quantile", alpha=q, rounds=500).fit(tr).predict(va)
        out["cutoff"] = c
        preds.append(out)
        print(f"  backtest {c}: ensemble WAPE {wape(out['y_demand'], out['ensemble']):.4f}", flush=True)
    bt = pd.concat(preds, ignore_index=True)
    meta = P.meta
    for col in ["SKU_ID", "Region", "Product_Family", "Sales_Channel", "Customer_Name"]:
        bt[col] = meta[col].to_numpy()[bt["series"]]
    bt["Date"] = np.array(MONTHS)[bt["target_m"]]
    return bt


def metrics(bt: pd.DataFrame, models: list[str]) -> dict[str, pd.DataFrame]:
    def table(by):
        rows = []
        grp = bt.groupby(by) if by else [("all", bt)]
        for k, g in grp:
            for mdl in models:
                rows.append({**({by: k} if by else {"level": "all"}), "model": mdl,
                             "wape": wape(g["y_demand"], g[mdl]), "bias": bias(g["y_demand"], g[mdl]),
                             "actual": float(g["y_demand"].sum()), "n": len(g)})
        return pd.DataFrame(rows)

    out = {"overall": table(None), "by_cutoff": table("cutoff"), "by_family": table("Product_Family"),
           "by_region": table("Region"), "by_horizon": table("h"), "by_channel": table("Sales_Channel")}
    # aggregate-level accuracy (forecasts summed bottom-up)
    for lvl in ["Product_Family", "Region"]:
        agg = bt.groupby(["cutoff", "Date", lvl])[["y_demand"] + models].sum().reset_index()
        rows = [{"level": lvl, "model": m, "wape": wape(agg["y_demand"], agg[m]),
                 "bias": bias(agg["y_demand"], agg[m])} for m in models]
        out[f"aggregate_{lvl}"] = pd.DataFrame(rows)
    if "q10" in bt:
        q = bt.dropna(subset=["q10", "q90"])
        out["interval_coverage"] = pd.DataFrame([{
            "cutoff": q["cutoff"].iloc[0],
            "p10_p90_coverage": float(((q["y_demand"] >= q["q10"]) & (q["y_demand"] <= q["q90"])).mean()),
            "below_p10": float((q["y_demand"] < q["q10"]).mean()),
            "above_p90": float((q["y_demand"] > q["q90"]).mean()),
        }])
    return out
