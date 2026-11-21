"""Demand drivers: which macro indicator leads which family, at what lag; is the lag
structure stable; how is the shift to electric vehicles showing up in demand."""
from __future__ import annotations

import numpy as np
import pandas as pd

from celeris.features.build import MACRO

LEVEL_INDICATORS = {"ISM_Manufacturing_PMI", "Industrial_Util_Rate_Pct", "Auto_Production_YoY_Pct",
                    "Vehicle_Inventory_Days"}
EXCLUDE_SHOCKS = ("2020-01", "2021-12")  # COVID + rebound dominate any correlation


def _demand_growth(demand: pd.DataFrame, by: str) -> pd.DataFrame:
    """YoY log growth of the 3-month moving average (removes seasonality and level)."""
    tot = demand.groupby([by, "Date"])["demand"].sum().unstack(0).sort_index()
    ma = tot.rolling(3, min_periods=3).mean()
    return np.log(ma).diff(12)


def _indicator_transform(macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.set_index("Date")[MACRO].sort_index()
    out = pd.DataFrame(index=m.index)
    for c in MACRO:
        s = m[c].rolling(3, min_periods=1).mean()
        out[c] = s if c in LEVEL_INDICATORS else np.log(s).diff(12)
    return out


def lead_lag(demand: pd.DataFrame, macro: pd.DataFrame, by: str = "Product_Family",
             max_lag: int = 12, exclude: tuple[str, str] | None = EXCLUDE_SHOCKS,
             window: tuple[str, str] | None = None) -> pd.DataFrame:
    g = _demand_growth(demand, by)
    ind = _indicator_transform(macro)
    rows = []
    for fam in g.columns:
        y = g[fam]
        for c in MACRO:
            for lag in range(max_lag + 1):
                x = ind[c].shift(lag).reindex(y.index)
                mask = y.notna() & x.notna()
                if exclude:
                    mask &= ~y.index.to_series().between(*exclude)
                if window:
                    mask &= y.index.to_series().between(*window)
                if mask.sum() < 18:
                    continue
                r = np.corrcoef(y[mask], x[mask])[0, 1]
                rows.append({by: fam, "indicator": c, "lag": lag, "corr": r, "n": int(mask.sum())})
    return pd.DataFrame(rows)


def best_lags(ll: pd.DataFrame, by: str = "Product_Family") -> pd.DataFrame:
    idx = ll.groupby([by, "indicator"])["corr"].apply(lambda s: s.abs().idxmax())
    b = ll.loc[idx].assign(abs_corr=lambda d: d["corr"].abs())
    return b.sort_values([by, "abs_corr"], ascending=[True, False]).reset_index(drop=True)


def lag_stability(demand: pd.DataFrame, macro: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    """Best lag per family x indicator in rolling multi-year windows (excluding COVID)."""
    windows = [("2019-01", "2021-12"), ("2022-01", "2024-06"), ("2023-07", "2026-10")]
    out = []
    for lo, hi in windows:
        ll = lead_lag(demand, macro, window=(lo, hi), exclude=None if lo >= "2022" else EXCLUDE_SHOCKS)
        if ll.empty:
            continue
        b = best_lags(ll)
        b["window"] = f"{lo}..{hi}"
        out.append(b)
    return pd.concat(out, ignore_index=True)


def ev_shift(demand: pd.DataFrame, master: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = demand.merge(master[["SKU_ID", "Powertrain_Application"]], on="SKU_ID", how="left")
    by = d.groupby(["Date", "Powertrain_Application"])["demand"].sum().unstack()
    share = by.div(by.sum(axis=1), axis=0)
    ev = macro.set_index("Date")["EV_Adoption_Index"].reindex(by.index)
    tab = by.add_suffix("_units").join(share.add_suffix("_share")).join(ev).reset_index()
    yr = by.groupby(by.index.str[:4]).sum()
    growth = (yr.pct_change() * 100).round(1)
    summary = {
        "share_first_12m": share.iloc[:12].mean().round(3).to_dict(),
        "share_last_12m": share.iloc[-12:].mean().round(3).to_dict(),
        "yoy_growth_pct_by_year": growth.dropna(how="all").to_dict(orient="index"),
        "corr_share_vs_ev_index": {c: round(float(share[c].corr(ev)), 3) for c in share.columns},
    }
    return tab, summary
