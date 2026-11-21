"""Forecast intervals for aggregates (family, region, total).

Summing SKU-level P10/P90 gives bands that are far too wide (errors partly cancel when
series are added up). Instead, the backtest gives the ratio actual / forecast for each
aggregate, origin and month; its empirical 10th / 90th percentiles by horizon bucket,
divided by the median ratio, are applied to the bottom-up P50 of the final forecast.

Dividing by the median keeps the band centred on the P50: the backtest under-forecast
at older origins (bias -6.5% / -8.1% / -3.4% / -0.6% across the four cutoffs) but the
bias has shrunk to ~0 at the most recent one, so only the spread is carried forward,
not the old bias. The P50 itself stays bottom-up, so aggregates add up to the SKU rows.
"""
from __future__ import annotations

import pandas as pd

from celeris.config import TRAIN_END

LEVELS = {"Product_Family": ["Product_Family"], "Region": ["Region"], "Total": []}
H_BINS, H_LABELS = [0, 6, 12, 19], ["1-6", "7-12", "13-19"]


def _hbucket(h) -> pd.Series:
    return pd.cut(pd.Series(h), H_BINS, labels=H_LABELS).astype(str)


def interval_factors(bt: pd.DataFrame, model: str = "gbm_tweedie") -> pd.DataFrame:
    """Quantiles of actual / forecast for aggregate series in the backtest."""
    bt = bt.assign(h_bucket=_hbucket(bt["h"]).to_numpy())
    rows = []
    for level, keys in LEVELS.items():
        a = bt.groupby(["cutoff", "Date", "h_bucket", *keys])[["y_demand", model]].sum().reset_index()
        a = a[a[model] > 0]
        r = a["y_demand"] / a[model]
        q = r.groupby(a["h_bucket"]).quantile([0.1, 0.5, 0.9]).unstack()
        for hb, v in q.iterrows():
            rows.append({"level": level, "h_bucket": hb, "ratio_p10": v[0.1], "ratio_p50": v[0.5],
                         "ratio_p90": v[0.9], "n": int((a["h_bucket"] == hb).sum())})
    return pd.DataFrame(rows)


def aggregate_forecast(fc: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Bottom-up P50 per aggregate and month with calibrated P10 / P90."""
    origin = pd.Period(TRAIN_END, "M")
    out = []
    for level, keys in LEVELS.items():
        a = fc.groupby(["Date", *keys])[["p50", "Revenue_P50_USD"]].sum().reset_index()
        a["level"] = level
        a["member"] = a[keys[0]] if keys else "All"
        a["h"] = [(pd.Period(d, "M") - origin).n for d in a["Date"]]
        a["h_bucket"] = _hbucket(a["h"]).to_numpy()
        f = factors[factors["level"] == level].set_index("h_bucket")
        lo = (f["ratio_p10"] / f["ratio_p50"]).clip(upper=1.0)
        hi = (f["ratio_p90"] / f["ratio_p50"]).clip(lower=1.0)
        a["p10"] = a["p50"] * a["h_bucket"].map(lo)
        a["p90"] = a["p50"] * a["h_bucket"].map(hi)
        out.append(a[["level", "member", "Date", "h", "p10", "p50", "p90", "Revenue_P50_USD"]])
    return pd.concat(out, ignore_index=True)
