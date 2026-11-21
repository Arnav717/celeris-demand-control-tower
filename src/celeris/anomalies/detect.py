"""Outlier detection on demand, attribution to events, structural / account changes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from celeris.anomalies.events import match_events

KEY = ["SKU_ID", "Region"]


# ----------------------------------------------------------------------------- cycles
def ordering_cycle(units: pd.Series) -> float:
    """Typical months between orders (mode of gaps between non-zero months)."""
    nz = np.flatnonzero(units.fillna(0).to_numpy() > 0)
    if len(nz) < 3:
        return np.nan
    gaps = np.diff(nz)
    vals, cnt = np.unique(gaps, return_counts=True)
    return float(vals[cnt.argmax()])


def cycles_by_year(p: pd.DataFrame) -> pd.DataFrame:
    c = (p.groupby(KEY + ["Year"])["demand"].apply(ordering_cycle).rename("cycle").reset_index())
    return c


# ----------------------------------------------------------------------------- outliers
def detect_outliers(p: pd.DataFrame, events_monthly: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(KEY + ["Date"]).copy()
    g = p.groupby(KEY)
    nz = p["demand"].where(p["demand"] > 0)
    p["level"] = nz.groupby([p["SKU_ID"], p["Region"]]).transform(
        lambda s: s.rolling(13, center=True, min_periods=3).median())
    p["zero_share"] = g["demand"].transform(lambda s: (s == 0).mean())
    # Regular seasonality (incl. fixed holidays such as EU August shutdowns) is not an
    # anomaly: expected = local level x family-region-month seasonal index.
    # Series-level index (zeros included, so a customer that never orders in August has
    # index ~0) when >= 4 years exist; otherwise the family-region index.
    calm = ~p["Date"].between("2020-02", "2020-09")
    r_all = (p["demand"] / p["level"]).where(calm)
    keys_s = [p["SKU_ID"], p["Region"], p["Month"]]
    seas_s = r_all.groupby(keys_s).transform("median")
    n_s = r_all.groupby(keys_s).transform("count")
    base = (p["demand"] > 0) & (p["zero_share"] <= 0.15) & calm
    seas_f = r_all.where(base).groupby([p["Product_Family"], p["Region"], p["Month"]]).transform("median")
    p["seasonal_idx"] = seas_s.where(n_s >= 4, seas_f).fillna(seas_f).fillna(1.0)
    p["level"] = p["level"] * p["seasonal_idx"]
    lr = np.log((p["demand"] + 1) / (p["level"] + 1))
    mad = lr.where(p["demand"] > 0).groupby([p["SKU_ID"], p["Region"]]).transform(
        lambda s: (s - s.median()).abs().median())
    p["z"] = lr / (1.4826 * mad.clip(lower=0.05))
    p["ratio"] = p["demand"] / p["level"]

    spike = (p["ratio"] >= 2.0) & (p["z"] >= 4) & ((p["demand"] - p["level"]) >= 25)
    cyclic = p["zero_share"] > 0.15
    dip = (~cyclic) & (p["ratio"] <= 0.4) & (p["level"] >= 10) & (p["z"] <= -3) \
        & p["demand_source"].isin(["observed", "backorder_removed"])
    p["outlier"] = np.select([spike, dip], ["spike", "dip"], "")
    flagged = p[p["outlier"] != ""].copy()
    ev = match_events(flagged, events_monthly)
    flagged = flagged.join(ev)
    nxt = g["ratio"].shift(-1).reindex(flagged.index)
    flagged["next_ratio"] = nxt

    t = flagged["event_types"].fillna("")
    m = flagged["Month"]
    is_spike = flagged["outlier"] == "spike"
    conds = [
        is_spike & t.str.contains("Price Change|Announcement") & m.isin([11, 12]),
        is_spike & t.str.contains("Customer Event"),
        is_spike & t.str.contains("Market Entry"),
        is_spike,
        ~is_spike & t.str.contains("Disruption"),
        ~is_spike & t.str.contains("Customer Event"),
        ~is_spike & t.str.contains("Logistics"),
        ~is_spike & t.str.contains("Holiday"),
        ~is_spike & (nxt >= 1.5),
        ~is_spike,
    ]
    types = ["pre_price_rise_buying", "customer_event_uplift", "market_entry_ramp",
             "one_off_order", "covid_disruption", "customer_event_dip", "logistics_delay",
             "holiday_dip", "timing_shift_or_unlisted_stockout", "unexplained_dip"]
    actions = [
        "kept; modelled with price-rise feature (Nov/Dec before 1 Jan increase)",
        "kept; modelled as customer event",
        "kept; series start ramp",
        "kept in history; capped at local level in model target",
        "kept in history; down-weighted in model training",
        "kept; modelled as customer event",
        "kept; modelled with logistics event feature",
        "kept; modelled with holiday share feature",
        "kept; flagged for planner review (dip followed by catch-up)",
        "kept; flagged for planner review",
    ]
    flagged["anomaly_type"] = np.select(conds, types, "other")
    flagged["action"] = np.select(conds, actions, "")
    flagged["category"] = np.where(flagged["anomaly_type"].isin(
        ["timing_shift_or_unlisted_stockout", "unexplained_dip"]), "suspected", "real_event")
    flagged["reason"] = (
        flagged["outlier"] + ": demand " + flagged["demand"].round(0).astype("Int64").astype(str)
        + " vs local level " + flagged["level"].round(0).astype("Int64").astype(str)
        + " (x" + flagged["ratio"].round(2).astype(str) + ")"
        + np.where(flagged["event_desc"].notna(), " | " + flagged["event_desc"].fillna(""), "")
    )
    return p, flagged


def demand_model_target(p: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    """Model target: demand with one-off orders capped at local level; covid weight."""
    p = p.copy()
    p["demand_model"] = p["demand"]
    one_off = flagged[flagged["anomaly_type"] == "one_off_order"].index
    p.loc[one_off, "demand_model"] = p.loc[one_off, "level"].round()
    p["one_off_order"] = p.index.isin(one_off)
    cov = flagged[flagged["anomaly_type"] == "covid_disruption"]
    covid_months = set(zip(cov["Date"], cov["Region"]))
    p["covid_period"] = pd.Series(list(zip(p["Date"], p["Region"]))).isin(covid_months).values
    p["sample_weight"] = np.where(p["covid_period"], 0.3, 1.0)
    return p


# ----------------------------------------------------------------------------- structural
def family_breaks(p: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    import ruptures as rpt

    rows = []
    agg = p.groupby(["Product_Family", "Date"])["demand"].sum().unstack(0).sort_index()
    for fam in agg.columns:
        s = agg[fam].dropna()
        x = np.log(s.clip(lower=1))
        seas = x.groupby(s.index.str[5:]).transform("mean") - x.mean()
        y = (x - seas).to_numpy().reshape(-1, 1)
        bks = rpt.Pelt(model="l2", min_size=4).fit(y).predict(pen=np.var(y) * 4)[:-1]
        for b in bks:
            d = s.index[b]
            before, after = s.iloc[max(0, b - 6):b].mean(), s.iloc[b:b + 6].mean()
            lo, hi = str(pd.Period(d, "M") - 3), str(pd.Period(d, "M") + 1)
            near = events[(events["Start_Month"] <= hi) & (events["End_Month"] >= lo)
                          & events["Event_Type"].isin(["Disruption", "Supply Constraint", "Market Entry",
                                                       "Announcement", "Customer Event", "System"])]
            near = near[(near["Scope_Type"].isin(["Global", "Region"])) | (near["Scope_Value"] == fam)]
            desc = near.drop_duplicates("Description")
            rows.append({"Date": d, "Product_Family": fam, "level_before_6m": round(before),
                         "level_after_6m": round(after), "change_pct": round(100 * (after / before - 1), 1),
                         "nearby_events": "; ".join(desc["Description"].str[:60])})
    return pd.DataFrame(rows)


def series_level_shifts(p: pd.DataFrame, min_ratio: float = 2.0) -> pd.DataFrame:
    """Largest sustained 12m-vs-12m level change per series (account gains / losses)."""
    p = p.sort_values(KEY + ["Date"])
    g = p.groupby(KEY)["demand"]
    before = g.transform(lambda s: s.rolling(12, min_periods=9).mean().shift(1))
    after = g.transform(lambda s: s[::-1].rolling(12, min_periods=9).mean()[::-1])
    r = np.log((after + 1) / (before + 1))
    p = p.assign(_r=r, _b=before, _a=after)
    # Neither 12m window may overlap COVID (2020-02..2020-12) or, for Electrical
    # Assemblies, the semiconductor allocation (2021-03..2022-09).
    d = pd.PeriodIndex(p["Date"], freq="M")
    clear_covid = (d - 12 > pd.Period("2020-12", "M")) | (d + 11 < pd.Period("2020-02", "M"))
    clear_semi = (p["Product_Family"] != "Electrical Assemblies") | \
        (d - 12 > pd.Period("2022-09", "M")) | (d + 11 < pd.Period("2021-03", "M"))
    ok = (p[["_b", "_a"]].max(axis=1) >= 15) & clear_covid & clear_semi
    cand = p[ok & (p["_r"].abs() >= np.log(min_ratio))]
    best = cand.loc[cand.groupby(KEY)["_r"].apply(lambda s: s.abs().idxmax())]
    return best.assign(shift_ratio=np.exp(best["_r"]).round(2), level_before=best["_b"].round(1),
                       level_after=best["_a"].round(1))


def customer_switches(train: pd.DataFrame) -> pd.DataFrame:
    t = train.sort_values(KEY + ["Date"])
    prev = t.groupby(KEY)["Customer_Name"].shift()
    sw = t[prev.notna() & prev.ne(t["Customer_Name"])].assign(previous_customer=prev)
    return sw[["Date", "SKU_ID", "Region", "Product_Family", "Sales_Channel", "previous_customer", "Customer_Name"]]


def cycle_changes(cyc: pd.DataFrame) -> pd.DataFrame:
    c = cyc.dropna().sort_values(KEY + ["Year"])
    prev = c.groupby(KEY)["cycle"].shift()
    ch = c[prev.notna() & prev.ne(c["cycle"])].assign(previous_cycle=prev)
    # keep persistent changes only: the new cycle also holds the following year (or last year)
    nxt = c.groupby(KEY)["cycle"].shift(-1).reindex(ch.index)
    return ch[nxt.isna() | nxt.eq(ch["cycle"])]


def lost_accounts(p: pd.DataFrame, test: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    m = master.set_index("SKU_ID")
    last = p[p["Units_Sold"].fillna(0) > 0].groupby(KEY)["Date"].max().rename("last_order")
    in_test = set(zip(test["SKU_ID"], test["Region"]))
    rows = []
    for (sku, reg), lo in last.items():
        eol = m.at[sku, "EOL_Date"]
        ended_by_eol = isinstance(eol, str) and eol <= "2026-10"
        if (sku, reg) not in in_test and not ended_by_eol:
            rows.append({"SKU_ID": sku, "Region": reg, "Date": lo, "status": "absent_from_test (lost account)"})
        elif lo <= "2026-04" and not ended_by_eol:
            rows.append({"SKU_ID": sku, "Region": reg, "Date": lo, "status": "no orders in last 6 months"})
    return pd.DataFrame(rows)
