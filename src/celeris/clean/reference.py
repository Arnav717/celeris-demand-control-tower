"""Cleaners for the reference files: sku_master, macro, stockouts, events, test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from celeris.clean.dq import DQLog
from celeris.clean.labels import norm_region, norm_sku, parse_month

MACRO_COLS = [
    "ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
    "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
    "Fuel_Price_Index", "EV_Adoption_Index",
]


def clean_master(m: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    F = "sku_master.csv"
    log.count(F, "raw_rows", len(m))
    m = m.copy()
    m["SKU_ID"] = m["SKU_ID"].map(norm_sku)
    # Rule: each family belongs to exactly one BU. Majority BU per family wins.
    fam_bu = m.groupby("Product_Family")["Business_Unit"].agg(lambda s: s.mode().iloc[0])
    wrong = m["Business_Unit"].ne(m["Product_Family"].map(fam_bu))
    log.add(F, "business_unit_contradicts_family", wrong,
            "set to the family's BU (transactions and test.csv agree)",
            examples=m.loc[wrong, ["SKU_ID", "Product_Family", "Business_Unit"]])
    m.loc[wrong, "Business_Unit"] = m.loc[wrong, "Product_Family"].map(fam_bu)
    for c in ["Launch_Date", "EOL_Date"]:
        m[c] = m[c].map(parse_month)
    bad_status = (
        ((m["Lifecycle_Status"] == "Active") & m["EOL_Date"].notna())
        | ((m["Lifecycle_Status"] == "Planned Launch") & (m["Launch_Date"] <= "2026-10"))
    )
    log.add(F, "lifecycle_status_inconsistent", bad_status, "flagged", category="info")
    log.count(F, "clean_rows", len(m))
    return m


def clean_macro(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    F = "macro_indicators.csv"
    log.count(F, "raw_rows", len(df))
    df = df.copy()
    df["Date"] = df["Date"].map(parse_month)
    dup = df["Date"].duplicated(keep=False)
    log.add(F, "duplicate_month", df["Date"].duplicated(), "averaged the duplicate rows",
            detail="two different rows for the same month, no way to tell which is right",
            examples=df[dup])
    df = df.groupby("Date", as_index=False)[MACRO_COLS].mean()
    df["dq_flags"] = ""

    # PMI is bounded ~[30, 70]; a value < 20 is a decimal shift (4.79 -> 47.9).
    pmi = df["ISM_Manufacturing_PMI"]
    shift = pmi < 20
    log.add(F, "pmi_decimal_shift", shift, "multiplied by 10", examples=df.loc[shift, ["Date", "ISM_Manufacturing_PMI"]])
    df.loc[shift, "ISM_Manufacturing_PMI"] = pmi[shift] * 10
    df.loc[shift, "dq_flags"] += "pmi_x10"

    # Auto production YoY: a run of >= 3 consecutive |x| < 0.1 values is stored as a
    # fraction, not percent (2019: -0.015 next to 2020: -6.4).
    yoy = df["Auto_Production_YoY_Pct"]
    small = yoy.abs() < 0.1
    run_id = (small != small.shift()).cumsum()
    run_len = small.groupby(run_id).transform("sum")
    frac = small & (run_len >= 3)
    log.add(F, "yoy_stored_as_fraction", frac, "multiplied by 100",
            examples=df.loc[frac, ["Date", "Auto_Production_YoY_Pct"]])
    df.loc[frac, "Auto_Production_YoY_Pct"] = yoy[frac] * 100
    df.loc[frac, "dq_flags"] += "yoy_x100"

    na = df[MACRO_COLS].isna()
    log.add(F, "missing_value", int(na.values.sum()), "linear interpolation in time",
            examples=[{"Date": df["Date"].iloc[i], "column": MACRO_COLS[j]}
                      for i, j in zip(*np.where(na.values))])
    df[MACRO_COLS] = df[MACRO_COLS].interpolate(limit_direction="both")
    full = pd.period_range(df["Date"].min(), df["Date"].max(), freq="M").astype(str)
    missing_months = set(full) - set(df["Date"])
    log.add(F, "missing_month", len(missing_months), "interpolated")
    df = df.set_index("Date").reindex(full).rename_axis("Date")
    df[MACRO_COLS] = df[MACRO_COLS].interpolate(limit_direction="both")
    df["dq_flags"] = df["dq_flags"].fillna("missing_month")
    df = df.reset_index()
    df["Year"] = df["Date"].str[:4].astype(int)
    df["Month"] = df["Date"].str[5:].astype(int)
    log.count(F, "clean_rows", len(df))
    return df


def clean_stockouts(s: pd.DataFrame, train: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    F = "inventory_stockouts.csv"
    log.count(F, "raw_rows", len(s))
    s = s.copy()
    s["Date"] = s["Date"].map(parse_month)
    s["SKU_ID"] = s["SKU_ID"].map(norm_sku)
    s["Region"] = s["Region"].map(norm_region).fillna(s["Region"])
    dup = s.duplicated(["Date", "SKU_ID", "Region"])
    log.add(F, "duplicate_key", dup, "dropped")
    s = s[~dup]
    ident = (s["Requested_Units"] + s["Carried_In_Units"] - s["Shipped_Units"]
             - s["Backordered_Units"] - s["Lost_Units"])
    log.add(F, "lost_units_identity_broken", ident.ne(0), "flagged")

    # Reconcile against cleaned train: Shipped_Units should equal Units_Sold.
    j = s.merge(train[["Date", "SKU_ID", "Region", "Units_Sold"]], on=["Date", "SKU_ID", "Region"], how="left")
    no_row = j["Units_Sold"].isna()
    diff = ~no_row & j["Units_Sold"].ne(j["Shipped_Units"])
    log.add(F, "stockout_without_train_row", no_row, "kept; demand row created from Requested_Units",
            category="info")
    log.add(F, "shipped_ne_units_sold", diff, "flagged; Shipped_Units trusted for constrained history",
            examples=j.loc[diff, ["Date", "SKU_ID", "Region", "Shipped_Units", "Units_Sold"]])
    s = j.rename(columns={"Units_Sold": "Train_Units_Sold"})
    log.count(F, "clean_rows", len(s))
    return s


def clean_events(e: pd.DataFrame, log: DQLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events, monthly expansion). Open-ended events run to the end of the test window."""
    F = "events_calendar.csv"
    log.count(F, "raw_rows", len(e))
    e = e.copy()
    e["Start_Month"] = e["Start_Date"].map(parse_month)
    e["End_Month"] = e["End_Date"].map(parse_month)
    open_ended = e["End_Month"].isna()
    log.add(F, "open_ended_event", open_ended, "treated as running to 2028-05 (permanent effect)",
            category="info", examples=e.loc[open_ended, ["Event_ID", "Description"]])
    e["End_Month"] = e["End_Month"].fillna("2028-05")
    # Day-level dates: share of the month covered (holidays partly in a month).
    rows = []
    for _, r in e.iterrows():
        sd = pd.to_datetime(r["Start_Date"]) if len(str(r["Start_Date"])) > 7 else None
        ed = pd.to_datetime(r["End_Date"]) if isinstance(r["End_Date"], str) and len(r["End_Date"]) > 7 else None
        for p in pd.period_range(r["Start_Month"], r["End_Month"], freq="M"):
            frac = 1.0
            if sd is not None and ed is not None:
                lo = max(sd, p.start_time)
                hi = min(ed, p.end_time.normalize())
                frac = max(0, (hi - lo).days + 1) / p.days_in_month
            rows.append({"Event_ID": r["Event_ID"], "Date": str(p), "Scope_Type": r["Scope_Type"],
                         "Scope_Value": r["Scope_Value"], "Event_Type": r["Event_Type"],
                         "Description": r["Description"], "Month_Share": round(frac, 3)})
    monthly = pd.DataFrame(rows)
    log.count(F, "clean_rows", len(e))
    return e, monthly


def clean_test(t: pd.DataFrame, master: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    F = "test.csv"
    log.count(F, "raw_rows", len(t))
    t = t.copy()
    dup = t.duplicated(["Date", "SKU_ID", "Region"])
    log.add(F, "duplicate_key", dup, "kept (submission must keep every row)", category="info")
    m = master.set_index("SKU_ID")
    after_eol = t["SKU_ID"].map(m["EOL_Date"]).pipe(lambda s: s.notna() & (t["Date"] > s))
    before = t["Date"] < t["SKU_ID"].map(m["Launch_Date"])
    log.add(F, "row_after_eol", after_eol, "forecast forced to 0 (rule 5)", category="info")
    log.add(F, "row_before_launch", before, "forecast forced to 0 (rule 5)", category="info")
    log.count(F, "clean_rows", len(t))
    return t
