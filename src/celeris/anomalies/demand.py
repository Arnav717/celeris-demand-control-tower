"""Unconstrained demand estimation.

Units_Sold is constrained shipments. Demand (what customers requested for delivery in a
month) differs in:
  1. listed stockout months         -> Requested_Units (inventory_stockouts.csv)
  2. the month after a listed one   -> shipments include back-orders; subtract them
  3. unlisted months inside a known supply-constraint event (stockout file is incomplete)
                                     -> Units / fill-rate of listed rows (same family, month)
  4. missing rows / blank values    -> imputed from the series' local level
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SO_COLS = ["Requested_Units", "Carried_In_Units", "Shipped_Units", "Backordered_Units",
           "Lost_Units", "Fill_Rate_Pct", "Stockout_Days", "Reason"]


def _prev_month(d: pd.Series) -> pd.Series:
    return (pd.PeriodIndex(d, freq="M") - 1).astype(str)


def estimate_demand(panel: pd.DataFrame, stockouts: pd.DataFrame,
                    events_monthly: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    so = stockouts[["Date", "SKU_ID", "Region"] + SO_COLS].copy()
    p = p.merge(so, on=["Date", "SKU_ID", "Region"], how="left")
    listed = p["Requested_Units"].notna()
    p["stockout_listed"] = listed

    p["demand"] = p["Units_Sold"].astype(float)
    p["demand_source"] = np.where(p["Units_Sold"].notna(), "observed", "")

    # (c) stockout rows without a train row: shipped is known.
    no_row = listed & p["Units_Sold"].isna()
    p.loc[no_row, "Units_Sold"] = p.loc[no_row, "Shipped_Units"]

    # (1) listed months: true demand = requested
    p.loc[listed, "demand"] = p.loc[listed, "Requested_Units"]
    p.loc[listed, "demand_source"] = "stockout_requested"

    # (2) back-order catch-up in the following (unlisted) month
    bo = so[so["Backordered_Units"] > 0][["Date", "SKU_ID", "Region", "Backordered_Units"]].copy()
    bo["Date"] = (pd.PeriodIndex(bo["Date"], freq="M") + 1).astype(str)
    bo = bo.rename(columns={"Backordered_Units": "bo_prev"})
    p = p.merge(bo, on=["Date", "SKU_ID", "Region"], how="left")
    catch = p["bo_prev"].notna() & ~listed & p["Units_Sold"].notna()
    p.loc[catch, "demand"] = (p.loc[catch, "Units_Sold"] - p.loc[catch, "bo_prev"]).clip(lower=0)
    p.loc[catch, "demand_source"] = "backorder_removed"

    # (3) unlisted months inside a supply-constraint event for the family
    sc = events_monthly[events_monthly["Event_Type"].eq("Supply Constraint")
                        & events_monthly["Scope_Type"].eq("Product_Family")]
    sc_keys = set(zip(sc["Date"], sc["Scope_Value"]))
    in_sc = pd.Series(list(zip(p["Date"], p["Product_Family"]))).isin(sc_keys).values
    lst = p[listed & in_sc]
    fr = (lst.groupby(["Product_Family", "Date"])["Shipped_Units"].sum()
          / (lst.groupby(["Product_Family", "Date"])["Requested_Units"].sum()
             + lst.groupby(["Product_Family", "Date"])["Carried_In_Units"].sum()))
    fr = fr.clip(0.2, 1.0).rename("fill_rate_est")
    p = p.merge(fr.reset_index(), on=["Product_Family", "Date"], how="left")
    inferred = in_sc & ~listed & ~catch & p["Units_Sold"].notna() & p["fill_rate_est"].notna()
    p.loc[inferred, "demand"] = (p.loc[inferred, "Units_Sold"] / p.loc[inferred, "fill_rate_est"]).round()
    p.loc[inferred, "demand_source"] = "stockout_inferred"
    p["stockout_inferred"] = inferred

    # (4) impute missing rows / blank values from a centred local level
    p = p.sort_values(["SKU_ID", "Region", "Date"]).reset_index(drop=True)
    g = p.groupby(["SKU_ID", "Region"])["demand"]
    local = g.transform(lambda s: s.rolling(7, center=True, min_periods=2).median())
    wider = g.transform(lambda s: s.rolling(13, center=True, min_periods=1).median())
    fill = local.fillna(wider)
    miss = p["demand"].isna() & p["row_status"].isin(["missing_row", "missing_value"])
    p.loc[miss, "demand"] = fill[miss].round()
    p.loc[miss, "demand_source"] = "imputed_" + p.loc[miss, "missing_reason"]
    # After the last order: do not invent demand (possible lost account) -> stays NaN.
    p["demand_imputed"] = miss
    p["lost_units_est"] = (p["demand"] - p["Units_Sold"]).where(listed | inferred).clip(lower=0)
    p = p.drop(columns=["bo_prev"])
    return p
