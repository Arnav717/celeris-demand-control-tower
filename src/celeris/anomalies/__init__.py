"""Anomaly stage: unconstrained demand + anomaly register.

Outputs (data/processed):
  demand.parquet            panel + demand, demand_model, flags, weights
  anomaly_register.parquet  one row per flagged point / series / family event
  family_breaks.parquet     structural breaks in family totals
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from celeris.anomalies.demand import estimate_demand
from celeris.anomalies.detect import (
    customer_switches,
    cycle_changes,
    cycles_by_year,
    demand_model_target,
    detect_outliers,
    family_breaks,
    lost_accounts,
    series_level_shifts,
)
from celeris.config import OUTPUTS, PROCESSED, REPORTS

REG_COLS = ["Date", "SKU_ID", "Region", "Product_Family", "Customer_Name", "Sales_Channel",
            "scope", "anomaly_type", "category", "value", "expected", "reason", "action", "event_ids"]

DQ_ACTIONS = {
    "price_x100": ("price entered in cents (x100)", "price set to SKU-year list price"),
    "price_x10": ("price per pack of 10", "price set to SKU-year list price"),
    "units_uom_x10": ("units booked in packs of 10", "units = revenue / list price"),
    "price_currency_eur": ("price & revenue booked in EUR", "converted to USD list price"),
    "price_mismatch": ("price differs from SKU-year list price", "price set to list price"),
    "price_missing": ("price 0 on a row with units", "price set to list price"),
    "units_rescaled": ("units x10/x100 vs revenue", "units rescaled to match revenue"),
    "revenue_rescaled": ("revenue x10/x100 vs units", "revenue rebuilt"),
    "units_sign": ("negative units (sign error)", "sign flipped"),
    "units_from_revenue": ("units missing", "units = revenue / price"),
    "units_missing": ("units and revenue missing", "demand imputed"),
    "units_from_stockout": ("units disagree with stockout Shipped_Units", "set to Shipped_Units"),
    "date_repaired": ("impossible date (e.g. 2062)", "date from load timestamp"),
    "region_inferred": ("region blank", "inferred from series"),
}


def _ctx(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Product_Family", "Customer_Name", "Sales_Channel"]:
        if c not in df:
            df[c] = None
    return df


def build_register(p, flagged, train, fam_brk, shifts, switches, cyc_ch, lost) -> pd.DataFrame:
    parts = []

    # 1) data errors fixed during cleaning (row level)
    t = train[train["dq_flags"] != ""][["Date", "SKU_ID", "Region", "Product_Family", "Customer_Name",
                                       "Sales_Channel", "Units_raw", "Units_Sold", "dq_flags"]]
    t = t.assign(flag=t["dq_flags"].str.split("|")).explode("flag")
    t = t[t["flag"].isin(DQ_ACTIONS)]
    parts.append(t.assign(
        scope="point", anomaly_type="error_" + t["flag"], category="error",
        value=t["Units_raw"], expected=t["Units_Sold"],
        reason=t["flag"].map(lambda f: DQ_ACTIONS[f][0]),
        action=t["flag"].map(lambda f: DQ_ACTIONS[f][1]), event_ids=None))

    # 2) demand corrections (stockouts / gaps)
    src = p[~p["demand_source"].isin(["observed", ""])]
    src_reason = {
        "stockout_requested": ("listed stockout: shipped < requested", "demand = Requested_Units"),
        "backorder_removed": ("month after stockout ships back-orders", "demand = shipped - back-orders"),
        "stockout_inferred": ("unlisted month inside supply-constraint event", "demand = shipped / family fill rate"),
    }
    def rs(s, i):
        return src_reason.get(s, (f"{s.replace('imputed_', '')} (no usable units)", "demand imputed from local level"))[i]
    parts.append(src.assign(
        scope="point",
        anomaly_type=src["demand_source"].where(~src["demand_source"].str.startswith("imputed"), "missing_data"),
        category=np.where(src["demand_source"].str.startswith("imputed"), "data_gap", "stockout"),
        value=src["Units_Sold"], expected=src["demand"],
        reason=[rs(s, 0) + (f" ({r})" if isinstance(r, str) and r else "") for s, r in zip(src["demand_source"], src["Reason"])],
        action=[rs(s, 1) for s in src["demand_source"]], event_ids=None))

    # 3) outliers on demand
    parts.append(flagged.assign(scope="point", value=flagged["demand"], expected=flagged["level"].round()))

    # 4) structural / account level
    if len(fam_brk):
        parts.append(_ctx(fam_brk.assign(SKU_ID=None, Region=None)).assign(
            scope="family", anomaly_type="structural_break", category="real_event",
            value=fam_brk["level_after_6m"], expected=fam_brk["level_before_6m"],
            reason="family level change " + fam_brk["change_pct"].astype(str) + "% | " + fam_brk["nearby_events"],
            action="model uses recent level / event features", event_ids=None))
    if len(shifts):
        parts.append(shifts.assign(
            scope="series", anomaly_type=np.where(shifts["shift_ratio"] > 1, "level_shift_up", "level_shift_down"),
            category="real_event", value=shifts["level_after"], expected=shifts["level_before"],
            reason="12m mean x" + shifts["shift_ratio"].astype(str),
            action="forecast anchored on post-shift level", event_ids=None))
    if len(switches):
        parts.append(switches.assign(
            scope="series", anomaly_type="customer_switch", category="real_event", value=None, expected=None,
            reason="customer changed from " + switches["previous_customer"] + " to " + switches["Customer_Name"],
            action="history kept; forecast for current customer", event_ids=None))
    if len(cyc_ch):
        c = _ctx(cyc_ch.assign(Date=cyc_ch["Year"].astype(str) + "-01"))
        parts.append(c.assign(
            scope="series", anomaly_type="ordering_cycle_change", category="real_event",
            value=c["cycle"], expected=c["previous_cycle"],
            reason="ordering cycle " + c["previous_cycle"].astype(int).astype(str) + "m -> "
                   + c["cycle"].astype(int).astype(str) + "m",
            action="recent cycle used in features", event_ids=None))
    if len(lost):
        parts.append(_ctx(lost).assign(
            scope="series", anomaly_type="lost_account", category="real_event", value=None, expected=None,
            reason=lost["status"], action="excluded from forecast / flagged", event_ids=None))

    reg = pd.concat([_ctx(x)[REG_COLS] for x in parts], ignore_index=True)
    reg["value"] = pd.to_numeric(reg["value"], errors="coerce")
    reg["expected"] = pd.to_numeric(reg["expected"], errors="coerce")
    return reg.sort_values(["Date", "SKU_ID", "Region"], na_position="first").reset_index(drop=True)


def price_rise_buying(p: pd.DataFrame) -> pd.DataFrame:
    """Nov+Dec vs Sep+Oct and Jan+Feb vs Sep+Oct at total level, by year."""
    tot = p.groupby("Date")["demand"].sum()
    rows = []
    for y in range(2019, 2026):
        so = tot.get(f"{y}-09", np.nan) + tot.get(f"{y}-10", np.nan)
        nd = tot.get(f"{y}-11", np.nan) + tot.get(f"{y}-12", np.nan)
        jf = tot.get(f"{y + 1}-01", np.nan) + tot.get(f"{y + 1}-02", np.nan)
        rows.append({"year": y, "novdec_vs_sepoct": round(nd / so, 3), "next_janfeb_vs_sepoct": round(jf / so, 3)})
    return pd.DataFrame(rows)


def run(out_dir=PROCESSED) -> None:
    panel = pd.read_parquet(out_dir / "panel.parquet")
    stock = pd.read_parquet(out_dir / "stockouts.parquet")
    em = pd.read_parquet(out_dir / "events_monthly.parquet")
    events = pd.read_parquet(out_dir / "events.parquet")
    train = pd.read_parquet(out_dir / "train_clean.parquet")
    test = pd.read_parquet(out_dir / "test.parquet")
    master = pd.read_parquet(out_dir / "sku_master.parquet")

    p = estimate_demand(panel, stock, em)
    p, flagged = detect_outliers(p, em)
    p = demand_model_target(p, flagged)
    cyc = cycles_by_year(p)
    fam_brk = family_breaks(p, events)
    shifts = series_level_shifts(p)
    switches = customer_switches(train)
    cyc_ch = cycle_changes(cyc)
    lost = lost_accounts(p, test, master)
    reg = build_register(p, flagged, train, fam_brk, shifts, switches, cyc_ch, lost)
    prb = price_rise_buying(p)

    p.to_parquet(out_dir / "demand.parquet", index=False)
    reg.to_parquet(out_dir / "anomaly_register.parquet", index=False)
    fam_brk.to_parquet(out_dir / "family_breaks.parquet", index=False)
    cyc.to_parquet(out_dir / "ordering_cycles.parquet", index=False)
    OUTPUTS.mkdir(exist_ok=True)
    reg.to_csv(OUTPUTS / "anomaly_register.csv", index=False)
    summary = {
        "register_rows": len(reg),
        "by_category": reg["category"].value_counts().to_dict(),
        "by_type": reg["anomaly_type"].value_counts().to_dict(),
        "demand_vs_shipments_uplift_pct": round(100 * (p["demand"].sum() / p["Units_Sold"].sum() - 1), 2),
        "lost_units_est": float(p["lost_units_est"].sum()),
        "price_rise_buying": prb.to_dict(orient="records"),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "anomaly_summary.json").write_text(json.dumps(summary, indent=2, default=str))
