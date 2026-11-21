"""Forecast stage: backtest, final model, lifecycle rules, submission.

Outputs:
  outputs/submission.csv                 Date, SKU_ID, Region, Units_Sold_Forecast
  data/processed/forecast.parquet        test rows + P10/P50/P90 + revenue
  data/processed/backtest_predictions.parquet
  reports/backtest_metrics.json          WAPE / bias tables
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from celeris.config import OUTPUTS, PROCESSED, REPORTS, TRAIN_END
from celeris.features.build import MIDX, MONTHS, Exog, Panel, build_frame
from celeris.models.backtest import CUTOFFS, GBM_SPECS, metrics, run_backtest
from celeris.models.gbm import GBM
from celeris.models.lifecycle import apply_lifecycle
from celeris.models.baselines import seasonal_index_level

TRAIN_ORIGIN_START = "2020-06"
ORIGIN_STRIDE = 2


def load(out_dir=PROCESSED):
    r = lambda n: pd.read_parquet(out_dir / f"{n}.parquet")  # noqa: E731
    return r("demand"), r("test"), r("sku_master"), r("macro"), r("events"), r("events_monthly")


def price_for(date: str, sku: str, list_2026: pd.Series, rise: dict[int, float]) -> float:
    p = list_2026.get(sku, np.nan)
    y = int(date[:4])
    for yy in range(2027, y + 1):
        p *= 1 + rise.get(yy, 0.0) / 100
    return p


def final_forecast(P, X, train_frame, test, master, demand) -> tuple[pd.DataFrame, dict]:
    t = MIDX[TRAIN_END]
    fr = build_frame(P, X, [t], require_target=False)
    # Point forecast: Tweedie GBM. In the backtest it matched the Tweedie+L1 ensemble on
    # WAPE (0.2745 vs 0.2749) with smaller bias, and was best at family level.
    models = {"gbm_tweedie": GBM(**GBM_SPECS["gbm_tweedie"]).fit(train_frame)}
    p50 = models["gbm_tweedie"].predict(fr)
    q10 = GBM("quantile", alpha=0.1, rounds=500).fit(train_frame).predict(fr)
    q90 = GBM("quantile", alpha=0.9, rounds=500).fit(train_frame).predict(fr)
    fr_out = pd.DataFrame({
        "SKU_ID": P.meta["SKU_ID"].to_numpy()[fr["series"]],
        "Region": P.meta["Region"].to_numpy()[fr["series"]],
        "Date": np.array(MONTHS)[fr["target_m"]],
        "p50_raw": p50, "p10": np.minimum(q10, p50), "p90": np.maximum(q90, p50), "source": "gbm",
    })
    fr_out["p50"] = fr_out["p50_raw"]
    fc = test[["Date", "SKU_ID", "Region", "Product_Family", "Business_Unit", "Customer_Name",
               "Sales_Channel"]].merge(fr_out, on=["Date", "SKU_ID", "Region"], how="left")

    # Fallback for series with < 6 months history: seasonal index x level baseline.
    miss = fc["p50"].isna()
    if miss.any():
        F = seasonal_index_level(P.Y, t)
        sidx = {k: i for i, k in enumerate(zip(P.meta["SKU_ID"], P.meta["Region"]))}
        vals = [F[sidx[(s, r)], MIDX[d] - t - 1] if (s, r) in sidx else 0.0
                for s, r, d in zip(fc.loc[miss, "SKU_ID"], fc.loc[miss, "Region"], fc.loc[miss, "Date"])]
        fc.loc[miss, "p50"] = fc.loc[miss, "p50_raw"] = vals
        fc.loc[miss, "p10"] = np.array(vals) * 0.5
        fc.loc[miss, "p90"] = np.array(vals) * 1.6
        fc.loc[miss, "source"] = "baseline_short_history"
    fc = apply_lifecycle(fc, master, demand)
    fc.loc[fc["lifecycle_rule"].str.startswith("new_launch"), "source"] = "new_launch_rule"
    for q in ("p10", "p50", "p90"):
        fc[q] = fc[q].clip(lower=0)
    fc["p10"] = np.minimum(fc["p10"], fc["p50"])
    fc["p90"] = np.maximum(fc["p90"], fc["p50"])
    fc["Units_Sold_Forecast"] = np.rint(fc["p50"]).astype(int)

    # revenue: 2026 list price x announced annual increases (EV-059, EV-065)
    lp = (demand.groupby("SKU_ID")["Unit_Price_USD"].last())
    rise = {y: v for y, v in X.rise.items()}
    fc["Unit_Price_USD"] = [price_for(d, s, lp, rise) for d, s in zip(fc["Date"], fc["SKU_ID"])]
    for s in fc.loc[fc["Unit_Price_USD"].isna(), "SKU_ID"].unique():  # new SKUs: predecessor price
        old = master.set_index("SKU_ID").at[s, "Replaces_SKU"]
        rows = fc["SKU_ID"] == s
        fc.loc[rows, "Unit_Price_USD"] = [price_for(d, old, lp, rise) for d in fc.loc[rows, "Date"]]
    fc["Revenue_P50_USD"] = fc["p50"] * fc["Unit_Price_USD"]
    imp = {n: m.importance().head(30).to_dict(orient="records") for n, m in models.items()}
    return fc, imp


def run(out_dir=PROCESSED) -> None:
    demand, test, master, macro, events, em = load(out_dir)
    P = Panel(demand, test, master)
    X = Exog(P, macro, events, em)
    origins = list(range(MIDX[TRAIN_ORIGIN_START], MIDX[TRAIN_END], ORIGIN_STRIDE))
    train_frame = build_frame(P, X, origins, max_target=MIDX[TRAIN_END])
    print(f"  train rows {len(train_frame):,}", flush=True)

    REPORTS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    if os.environ.get("CELERIS_SKIP_BACKTEST") != "1":
        bt = run_backtest(P, X, train_frame, CUTOFFS)
        bt.to_parquet(out_dir / "backtest_predictions.parquet", index=False)
        models = ["seasonal_naive", "moving_average_12", "seasonal_index_x_level", "tsb",
                  "gbm_tweedie", "gbm_l1", "ensemble"]
        mt = metrics(bt, models)
        for k, v in mt.items():
            v.to_parquet(out_dir / f"bt_{k}.parquet", index=False)
        (REPORTS / "backtest_metrics.json").write_text(json.dumps(
            {k: v.to_dict(orient="records") for k, v in mt.items()}, indent=2, default=float))
        print(mt["overall"].to_string(index=False), flush=True)

    fc, imp = final_forecast(P, X, train_frame, test, master, demand)
    fc.to_parquet(out_dir / "forecast.parquet", index=False)
    sub = fc[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]]
    assert len(sub) == len(test) and sub["Units_Sold_Forecast"].ge(0).all()
    sub.to_csv(OUTPUTS / "submission.csv", index=False)
    (REPORTS / "feature_importance.json").write_text(json.dumps(imp, indent=2, default=float))
    print(f"  submission rows {len(sub):,}; total P50 units {fc['p50'].sum():,.0f}", flush=True)
