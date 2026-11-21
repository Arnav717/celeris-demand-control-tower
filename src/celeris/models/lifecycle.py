"""Lifecycle rules for the forecast window.

* Rule 5: no demand after EOL_Date or before Launch_Date -> 0.
* EOL run-down: predecessors fall to ~0.5x in their last 3 months before EOL and
  ~0.8x in the 3 months before that (median over the 6 historical successor pairs).
* New launches with no history (Replaces_SKU): SKU-level volume = predecessor's
  pre-launch 12-month level x empirical ramp-up curve x predecessor seasonality, split
  across the new SKU's regions by family-region SKU size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from celeris.features.build import MIDX, MONTHS


def ramp_curve(demand: pd.DataFrame, master: pd.DataFrame, max_k: int = 24) -> pd.Series:
    """Median (over historical pairs) of new-SKU demand / predecessor pre-launch level."""
    tot = demand.groupby(["SKU_ID", "Date"])["demand"].sum().unstack(0)
    pairs = master[master["Replaces_SKU"].notna()]
    rows = []
    for _, r in pairs.iterrows():
        if r["SKU_ID"] not in tot or r["Replaces_SKU"] not in tot:
            continue
        L = pd.Period(r["Launch_Date"], "M")
        pre = tot[r["Replaces_SKU"]].loc[str(L - 12): str(L - 1)].mean()
        for k in range(max_k):
            mm = str(L + k)
            if mm in tot.index and pre > 0:
                rows.append({"k": k, "ratio": tot.at[mm, r["SKU_ID"]] / pre})
    c = pd.DataFrame(rows).groupby("k")["ratio"].median()
    c = c.rolling(3, center=True, min_periods=1).mean()
    plateau = c.loc[8:].median()
    c.loc[8:] = plateau
    return c.reindex(range(max_k)).ffill()


def rundown_multiplier(months_to_eol: np.ndarray) -> np.ndarray:
    m = np.ones_like(months_to_eol, dtype=float)
    m[(months_to_eol >= 0) & (months_to_eol <= 2)] = 0.5
    m[(months_to_eol >= 3) & (months_to_eol <= 5)] = 0.8
    return m


def apply_lifecycle(fc: pd.DataFrame, master: pd.DataFrame, demand: pd.DataFrame,
                    qcols=("p10", "p50", "p90")) -> pd.DataFrame:
    """fc: test rows with Date, SKU_ID, Region, Product_Family and quantile columns."""
    fc = fc.copy()
    m = master.set_index("SKU_ID")
    mi = fc["Date"].map(MIDX).to_numpy()
    eol = fc["SKU_ID"].map(m["EOL_Date"])
    launch = fc["SKU_ID"].map(m["Launch_Date"])
    eol_i = eol.map(lambda x: MIDX.get(x, 9999) if isinstance(x, str) else 9999).to_numpy()
    mte = eol_i - mi
    fc["lifecycle_rule"] = ""

    # EOL run-down + zero after EOL
    has_eol = eol.notna().to_numpy()
    mult = np.where(has_eol, rundown_multiplier(mte), 1.0)
    for q in qcols:
        fc[q] = fc[q] * mult
    fc.loc[has_eol & (mult < 1), "lifecycle_rule"] = "eol_rundown"
    after = has_eol & (mte < 0)
    before = (launch.notna() & (fc["Date"] < launch)).to_numpy()
    for q in qcols:
        fc.loc[after | before, q] = 0.0
    fc.loc[after, "lifecycle_rule"] = "after_eol_zero"
    fc.loc[before, "lifecycle_rule"] = "before_launch_zero"

    # New launches without history
    curve = ramp_curve(demand, master)
    hist_skus = set(demand["SKU_ID"])
    new = [s for s in fc["SKU_ID"].unique() if s not in hist_skus and isinstance(m.at[s, "Replaces_SKU"], str)]
    for s in new:
        old = m.at[s, "Replaces_SKU"]
        L = pd.Period(m.at[s, "Launch_Date"], "M")
        # predecessor pre-launch level (SKU total) from the forecast itself, else history
        old_fc = fc[fc["SKU_ID"] == old]
        pre_months = [str(L - k) for k in range(1, 13)]
        pre_fc = old_fc[old_fc["Date"].isin(pre_months)]
        hist = demand[demand["SKU_ID"] == old]
        if len(pre_fc):
            level = pre_fc.groupby("Date")["p50_raw"].sum().mean()
        else:
            level = hist.groupby("Date")["demand"].sum().iloc[-12:].mean()
        # predecessor seasonality (SKU total, last 3 years)
        h3 = hist[hist["Date"] >= str(pd.Period(hist["Date"].max(), "M") - 35)]
        tot = h3.groupby("Date")["demand"].sum()
        seas = tot.groupby(tot.index.str[5:]).mean() / tot.mean()
        # regional split by typical SKU size in the family-region
        fam = m.at[s, "Product_Family"]
        reg_size = (demand[demand["Product_Family"] == fam].groupby(["Region", "SKU_ID"])["demand"].mean()
                    .groupby("Region").median())
        rows = fc["SKU_ID"] == s
        regs = fc.loc[rows, "Region"].unique()
        share = reg_size.reindex(regs).fillna(reg_size.median())
        share = share / share.sum()
        for i in fc.index[rows]:
            d = fc.at[i, "Date"]
            k = (pd.Period(d, "M") - L).n
            if k < 0:
                continue
            v = level * curve.get(min(k, len(curve) - 1), curve.iloc[-1]) * seas.get(d[5:], 1.0) \
                * share[fc.at[i, "Region"]]
            fc.loc[i, "p50"] = v
            fc.loc[i, "p10"] = v * 0.4
            fc.loc[i, "p90"] = v * 1.8
            fc.loc[i, "lifecycle_rule"] = f"new_launch_from_{old}"
    return fc


def months_index(dates) -> np.ndarray:
    return np.array([MIDX[d] for d in dates])


__all__ = ["MONTHS", "apply_lifecycle", "ramp_curve", "rundown_multiplier"]
