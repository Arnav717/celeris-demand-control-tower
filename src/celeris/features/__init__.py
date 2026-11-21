"""Features & drivers stage.

The forecasting feature matrix is built on the fly (celeris.features.build). This stage
writes the driver analysis used in the write-up and dashboard:
  drivers_leadlag.parquet     corr(family demand growth, indicator lagged L months)
  drivers_best_lag.parquet    strongest lag per family x indicator
  drivers_stability.parquet   best lag per family x indicator in three time windows
  ev_shift.parquet            demand by powertrain (ICE / EV / Universal) vs EV index
  series_profile.parquet      intermittency class (ADI / CV2) and ordering cycle per series
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from celeris.config import PROCESSED, REPORTS
from celeris.features.drivers import best_lags, ev_shift, lag_stability, lead_lag


def intermittency(demand: pd.DataFrame, since: str = "2024-11") -> pd.DataFrame:
    """Syntetos-Boylan classes on the last 24 months: smooth / erratic / intermittent / lumpy."""
    d = demand[demand["Date"] >= since]

    def stats(s):
        s = s.dropna()
        nz = s[s > 0]
        adi = len(s) / max(len(nz), 1)
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
        return pd.Series({"adi": adi, "cv2": cv2, "mean": s.mean(), "zero_share": (s == 0).mean()})

    st = d.groupby(["SKU_ID", "Region"])["demand"].apply(stats).unstack().reset_index()
    st["intermittency_class"] = np.select(
        [(st["adi"] < 1.32) & (st["cv2"] < 0.49), (st["adi"] < 1.32), (st["cv2"] < 0.49)],
        ["smooth", "erratic", "intermittent"], "lumpy")
    return st


def run(out_dir=PROCESSED) -> None:
    demand = pd.read_parquet(out_dir / "demand.parquet")
    macro = pd.read_parquet(out_dir / "macro.parquet")
    master = pd.read_parquet(out_dir / "sku_master.parquet")
    cyc = pd.read_parquet(out_dir / "ordering_cycles.parquet")

    ll = lead_lag(demand, macro)
    bl = best_lags(ll)
    st = lag_stability(demand, macro)
    ev_tab, ev_sum = ev_shift(demand, master, macro)
    prof = intermittency(demand)
    last_cyc = cyc.dropna().sort_values("Year").groupby(["SKU_ID", "Region"])["cycle"].last()
    prof = prof.join(last_cyc, on=["SKU_ID", "Region"])

    ll.to_parquet(out_dir / "drivers_leadlag.parquet", index=False)
    bl.to_parquet(out_dir / "drivers_best_lag.parquet", index=False)
    st.to_parquet(out_dir / "drivers_stability.parquet", index=False)
    ev_tab.to_parquet(out_dir / "ev_shift.parquet", index=False)
    prof.to_parquet(out_dir / "series_profile.parquet", index=False)

    top = bl.loc[bl.groupby("Product_Family")["corr"].apply(lambda s: s.abs().idxmax())]
    piv = st.pivot_table(index=["Product_Family", "indicator"], columns="window", values="lag")
    stable_share = float((piv.max(axis=1) - piv.min(axis=1) <= 2).mean())
    summary = {
        "top_indicator_per_family": top[["Product_Family", "indicator", "lag", "corr"]].round(3).to_dict(orient="records"),
        "share_family_indicator_pairs_with_stable_lag_(+-2m)": round(stable_share, 3),
        "intermittency_classes": prof["intermittency_class"].value_counts().to_dict(),
        "ev_shift": ev_sum,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "drivers_summary.json").write_text(json.dumps(summary, indent=2, default=str))
