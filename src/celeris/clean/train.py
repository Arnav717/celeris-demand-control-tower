"""Clean train.csv -> one row per Date-SKU-Region with reconciled units/price/revenue.

Every fix is logged to DQLog and recorded per row in `dq_flags` (pipe-separated)
so the anomaly monitor can show what was changed and why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from celeris.clean.dq import DQLog
from celeris.clean.labels import (
    build_customer_map,
    norm_channel,
    norm_region,
    norm_sku,
    parse_month,
)
from celeris.config import SURCHARGES, TRAIN_END, TRAIN_START

F = "train.csv"


def _flag(df: pd.DataFrame, mask: pd.Series, flag: str) -> None:
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    df.loc[mask, "dq_flags"] = df.loc[mask, "dq_flags"].where(
        df.loc[mask, "dq_flags"] == "", df.loc[mask, "dq_flags"] + "|"
    ) + flag


def surcharge_factor(dates: pd.Series) -> pd.Series:
    f = pd.Series(1.0, index=dates.index)
    for start, end, k in SURCHARGES:
        f[(dates >= start) & (dates <= end)] = k
    return f


def _close(a: pd.Series, b: pd.Series, rel: float = 0.005, abs_: float = 1.0) -> pd.Series:
    return (a - b).abs() <= np.maximum(abs_, rel * b.abs())


def load_raw(path) -> pd.DataFrame:
    # keep_default_na=False: the region code "NA" (North America) must not become NaN.
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


# --------------------------------------------------------------------------- steps
def dedupe(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["Load_Timestamp"], errors="coerce")
    dup = df["Record_ID"].duplicated(keep=False)
    grp = df[dup].groupby("Record_ID")
    revised = grp["Units_Sold"].transform("nunique").gt(1) | grp["Revenue_USD"].transform("nunique").gt(1)
    n_revised = revised.groupby(df.loc[dup, "Record_ID"]).any().sum()
    n_exact = grp.ngroups - n_revised
    df = df.sort_values(["Record_ID", "_ts"]).drop_duplicates("Record_ID", keep="last")
    log.add(F, "duplicate_record_exact", n_exact, "dropped earlier copy (identical reload)")
    log.add(F, "duplicate_record_revised", n_revised,
            "kept row with latest Load_Timestamp (business rule 2)",
            detail="same Record_ID re-loaded later with different Units_Sold/Revenue_USD")
    return df


def fix_dates(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df["Date_raw"] = df["Date"]
    parsed = df["Date"].map(parse_month)
    fmt_bad = ~df["Date"].str.fullmatch(r"\d{4}-\d{2}", na=False)
    log.add(F, "date_nonstandard_format", fmt_bad & parsed.notna(),
            "parsed to YYYY-MM", detail="formats seen: MM/YYYY, YYYY/MM, Mon-YYYY, Mon YYYY, MM-YYYY, YYYY-MM-DD")
    # Rows are loaded on the month after the requested delivery month (verified: 100% of
    # plausible rows). Use that only to repair impossible dates such as 2062-01 (digit swap).
    from_load = (df["_ts"].dt.to_period("M") - 1).astype(str)
    bad = parsed.isna() | (parsed < TRAIN_START) | (parsed > TRAIN_END)
    log.add(F, "date_out_of_range", bad, "replaced by Load_Timestamp month - 1",
            detail="e.g. 2062-01 loaded 2026-02 (digit transposition of 2026)",
            examples=df.loc[bad, ["Record_ID", "Date", "Year", "Month", "Load_Timestamp"]])
    df["Date"] = parsed.where(~bad, from_load)
    _flag(df, bad, "date_repaired")
    ym = df["Date"].str[:4].astype(int), df["Date"].str[5:].astype(int)
    mism = (pd.to_numeric(df["Year"], errors="coerce") != ym[0]) | (pd.to_numeric(df["Month"], errors="coerce") != ym[1])
    log.add(F, "year_month_disagree_with_date", mism & ~bad, "Year/Month rebuilt from Date (rule 1)")
    df["Year"], df["Month"] = ym
    return df


def fix_keys(df: pd.DataFrame, master: pd.DataFrame, log: DQLog,
             customers: list[str] | None = None) -> pd.DataFrame:
    sku = df["SKU_ID"].map(norm_sku)
    changed = sku.ne(df["SKU_ID"])
    log.add(F, "sku_id_malformed", changed & sku.notna(), "normalised to AAA-NNN",
            detail="lower case, '_' separator, missing dash, missing zero padding, stray spaces",
            examples=pd.DataFrame({"raw": df.loc[changed, "SKU_ID"], "clean": sku[changed]}).drop_duplicates())
    unknown = sku.isna() | ~sku.isin(master["SKU_ID"])
    log.add(F, "sku_id_unknown", unknown, "dropped")
    df["SKU_ID"] = sku
    df = df[~unknown].copy()

    reg = df["Region"].map(norm_region)
    log.add(F, "region_label_variant", reg.notna() & reg.ne(df["Region"]), "mapped to canonical region",
            examples=pd.DataFrame({"raw": df.loc[reg.ne(df["Region"]), "Region"], "clean": reg}).dropna().drop_duplicates())
    df["Region"] = reg

    ch = df["Sales_Channel"].map(norm_channel)
    log.add(F, "channel_label_variant", ch.notna() & ch.ne(df["Sales_Channel"]), "mapped to canonical channel")
    log.add(F, "channel_unknown", ch.isna(), "left as-is", category="info")
    df["Sales_Channel"] = ch.fillna(df["Sales_Channel"])

    cmap = build_customer_map(df["Customer_Name"], customers)
    cust = df["Customer_Name"].map(cmap)
    typo = cust.ne(df["Customer_Name"])
    log.add(F, "customer_name_variant", typo, "mapped to canonical customer (case, spacing, Inc. suffix, typos)",
            examples=pd.DataFrame({"raw": df.loc[typo, "Customer_Name"], "clean": cust[typo]}).drop_duplicates())
    df["Customer_Name"] = cust

    # Master data is the reference for SKU attributes.
    m = master.set_index("SKU_ID")
    for col in ["Product_Family", "Business_Unit", "SKU_Description"]:
        ref = df["SKU_ID"].map(m[col])
        missing = df[col].isna()
        conflict = df[col].notna() & df[col].str.strip().ne(ref)
        log.add(F, f"{col.lower()}_missing", missing, "filled from sku_master")
        log.add(F, f"{col.lower()}_conflicts_master", conflict, "overwritten with sku_master value")
        df[col] = ref
    return df


def infer_missing_region(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    """Rows with blank Region: assign the only region where this SKU+customer(+channel)
    series exists and has no row in that month. Otherwise drop (cannot key the row)."""
    miss = df["Region"].isna()
    known = df[~miss]
    occupied = set(zip(known["Date"], known["SKU_ID"], known["Region"]))
    by_sc = known.groupby(["SKU_ID", "Customer_Name"])["Region"].agg(lambda s: sorted(set(s))).to_dict()
    by_scc = known.groupby(["SKU_ID", "Customer_Name", "Sales_Channel"])["Region"].agg(lambda s: sorted(set(s))).to_dict()
    by_s = known.groupby("SKU_ID")["Region"].agg(lambda s: sorted(set(s))).to_dict()
    assigned = {}
    for idx, r in df[miss].iterrows():
        for cands in (by_scc.get((r.SKU_ID, r.Customer_Name, r.Sales_Channel)),
                      by_sc.get((r.SKU_ID, r.Customer_Name)), by_s.get(r.SKU_ID)):
            if not cands:
                continue
            free = [c for c in cands if (r.Date, r.SKU_ID, c) not in occupied]
            if len(free) == 1:
                assigned[idx] = free[0]
                occupied.add((r.Date, r.SKU_ID, free[0]))
                break
            if len(free) > 1:
                break  # ambiguous at this level -> stop, do not guess more broadly
    s = pd.Series(assigned, dtype=object)
    df.loc[s.index, "Region"] = s
    _flag(df, df.index.isin(s.index), "region_inferred")
    log.add(F, "region_missing_inferred", len(s),
            "inferred from SKU+customer+channel series with a free slot that month")
    still = df["Region"].isna()
    log.add(F, "region_missing_unresolved", still, "dropped (ambiguous region)")
    return df[~still].copy()


def fix_price_units_revenue(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    units_txt = df["Units_Sold"]
    u = pd.to_numeric(units_txt.str.replace(",", "", regex=False).str.strip(), errors="coerce")
    log.add(F, "units_thousands_separator", units_txt.str.contains(",", na=False), "parsed as number")
    log.add(F, "units_placeholder_dash", units_txt.str.strip().eq("-"), "treated as missing value")
    p = pd.to_numeric(df["Unit_Price_USD"], errors="coerce")
    r = pd.to_numeric(df["Revenue_USD"], errors="coerce")
    df["Units_raw"], df["Price_raw"], df["Revenue_raw"] = u, p, r

    # ---- price: one list price per SKU per calendar year (rule 4) x surcharge.
    sf = surcharge_factor(df["Date"])
    base_rows = df[(p > 0) & (sf == 1.0)]
    base = (base_rows.assign(_p=(p / sf).round(2)).groupby(["SKU_ID", "Year"])["_p"]
            .agg(lambda s: s.mode().iloc[0]))
    # Fill SKU-years without a usable price from the neighbouring year.
    full = base.unstack().ffill(axis=1).bfill(axis=1).stack()
    list_price = pd.Series(list(zip(df["SKU_ID"], df["Year"])), index=df.index).map(full)
    p_star = (list_price * sf).round(2)
    ratio = p / p_star

    ok = (ratio - 1).abs() < 0.005
    cents = ((ratio / 100) - 1).abs() < 0.02
    x10 = ((ratio / 10) - 1).abs() < 0.02
    eur = ratio.between(0.80, 0.95)
    zero_price = p.fillna(0).eq(0)
    other = ~(ok | cents | x10 | eur | zero_price)
    log.add(F, "price_x100_cents", cents, "replaced with SKU-year list price",
            detail="price entered in cents (x100); revenue is correct in USD")
    log.add(F, "price_x10", x10, "replaced with SKU-year list price", detail="mostly India rows")
    log.add(F, "price_in_eur_not_usd", eur, "replaced with USD list price; revenue rebuilt",
            detail="Europe rows with price and revenue ~0.89x list (EUR booked as USD)",
            examples=df.loc[eur, ["Date", "SKU_ID", "Region", "Unit_Price_USD", "Revenue_USD"]])
    log.add(F, "price_zero_with_units", zero_price & u.fillna(0).ne(0), "replaced with SKU-year list price")
    log.add(F, "price_other_mismatch", other, "replaced with SKU-year list price")
    log.add(F, "price_surcharge_2022H2", (sf > 1) & ok, "kept (real: EV-027 6% surcharge)", category="real_event")
    for m_, name in [(cents, "price_x100"), (x10, "price_x10"), (eur, "price_currency_eur"),
                     (other, "price_mismatch"), (zero_price & u.fillna(0).ne(0), "price_missing")]:
        _flag(df, m_, name)

    # ---- units: reconcile with revenue.
    tol = lambda a, b: _close(a, b, rel=0.005, abs_=1.0)
    cons_raw = tol(u * p, r)          # consistent in booked currency/scale
    cons_star = tol(u * p_star, r)     # consistent with corrected USD price
    consistent = (cons_raw | cons_star) & u.notna() & r.notna()

    level = (df.assign(_u=u)[consistent & (u > 0)]
             .groupby(["SKU_ID", "Region"])["_u"].median())
    sku_level = df.assign(_u=u)[consistent & (u > 0)].groupby("SKU_ID")["_u"].median()
    lvl = pd.Series(list(zip(df["SKU_ID"], df["Region"])), index=df.index).map(level)
    lvl = lvl.fillna(df["SKU_ID"].map(sku_level)).fillna(u.median())

    units = u.copy()
    action = pd.Series("", index=df.index)

    # negative units: shipments cannot be negative; revenue carries the same sign -> sign error
    neg = u < 0
    units[neg] = u[neg].abs()
    action[neg] = "units_sign_flipped"
    log.add(F, "units_negative", neg, "sign flipped (abs)",
            detail="negative units with negative revenue of matching magnitude; |units| in series' normal range -> sign error, not returns")

    # missing units: rebuild from revenue / price when revenue is present
    miss = u.isna()
    from_rev_star = r / p_star
    from_rev_raw = r / p.replace(0, np.nan)
    pick = np.where((from_rev_star - from_rev_star.round()).abs() < 0.02, from_rev_star,
                    np.where((from_rev_raw - from_rev_raw.round()).abs() < 0.02, from_rev_raw, from_rev_star))
    rebuilt = pd.Series(pick, index=df.index).round()
    can = miss & r.notna() & rebuilt.notna()
    units[can] = rebuilt[can]
    action[can] = "units_from_revenue"
    log.add(F, "units_missing_rebuilt", can, "units = revenue / price")
    log.add(F, "units_missing_unrecoverable", miss & ~can,
            "left NaN (missing value, not zero); imputed in demand stage")

    # x10 price rows are a unit-of-measure error: quantity booked in packs of 10 and price
    # per pack, revenue correct. True units = revenue / list price (validated against
    # inventory_stockouts Shipped_Units and the series level).
    uom = x10 & cons_raw & u.notna() & ~neg
    units[uom] = (r[uom] / p_star[uom]).round()
    action[uom] = "units_uom_x10"
    log.add(F, "units_uom_packs_of_10", uom, "units = revenue / list price (x10)",
            detail="price x10 and units /10 with correct revenue; mostly India")
    _flag(df, uom, "units_uom_x10")

    # local level: centred 7-month median of reconciled rows in the same series
    loc = (df.assign(_u=u.where(consistent & ~x10 & (u > 0)))
           .sort_values("Date").groupby(["SKU_ID", "Region"])["_u"]
           .transform(lambda s: s.rolling(7, center=True, min_periods=2).median()))
    lvl = loc.reindex(df.index).fillna(lvl)

    # unit/revenue x10 disagreements: decide which side is wrong via the local level
    chk = ~miss & ~neg & ~consistent & r.notna() & (u != 0)
    k_star = r / (u * p_star)
    k_raw = r / (u * p)
    k = k_star.where(chk)
    for kk in (k_star, k_raw):
        for f in (10.0, 0.1, 100.0, 0.01):
            hit = chk & ((kk / f) - 1).abs().lt(0.02)
            k[hit] = f
    scale_hit = chk & k.isin([10.0, 0.1, 100.0, 0.01])
    alt = (u * k).round()
    units_wrong = scale_hit & ((np.log(alt.clip(lower=1)) - np.log(lvl)).abs()
                               < (np.log(u.clip(lower=1)) - np.log(lvl)).abs())
    units[units_wrong] = alt[units_wrong]
    action[units_wrong] = "units_rescaled"
    log.add(F, "units_scale_error", units_wrong, "units rescaled to match revenue",
            detail="units off by x10/x100 vs revenue; corrected value closer to series level")
    log.add(F, "revenue_scale_error", scale_hit & ~units_wrong, "revenue rebuilt = units x price",
            detail="revenue off by x10/x100; units consistent with series level")
    log.add(F, "revenue_minor_mismatch", chk & ~scale_hit, "revenue rebuilt = units x price",
            detail="units x price != revenue by a non-scale factor; units trusted")
    log.add(F, "revenue_missing", r.isna() & units.notna(), "revenue rebuilt = units x price")

    _flag(df, units_wrong, "units_rescaled")
    _flag(df, neg, "units_sign")
    _flag(df, can, "units_from_revenue")
    _flag(df, miss & ~can, "units_missing")
    _flag(df, scale_hit & ~units_wrong, "revenue_rescaled")

    df["Units_Sold"] = units
    df["Unit_Price_USD"] = p_star
    df["List_Price_USD"] = list_price
    df["Revenue_USD"] = (units * p_star).round(2)
    df["fix_units"] = action
    return df


def apply_lifecycle(df: pd.DataFrame, master: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    m = master.set_index("SKU_ID")
    launch = df["SKU_ID"].map(m["Launch_Date"])
    eol = df["SKU_ID"].map(m["EOL_Date"])
    after = eol.notna() & (df["Date"] > eol)
    before = launch.notna() & (df["Date"] < launch)
    pos_after = after & df["Units_Sold"].fillna(0).gt(0)
    log.add(F, "sales_after_eol", pos_after, "dropped (rule 5: no sales after EOL_Date)",
            examples=df.loc[pos_after, ["Date", "SKU_ID", "Region", "Units_Sold"]])
    log.add(F, "zero_rows_after_eol", after & ~pos_after, "dropped (no demand possible)")
    log.add(F, "sales_before_launch", before, "dropped (rule 5)",
            examples=df.loc[before, ["Date", "SKU_ID", "Region", "Units_Sold"]])
    return df[~(after | before)].copy()


def apply_stockout_shipped(df: pd.DataFrame, stockouts: pd.DataFrame | None, log: DQLog) -> pd.DataFrame:
    """inventory_stockouts.Shipped_Units is 'what train.csv should show as Units_Sold'.
    Where both exist and disagree after cleaning, trust the stockout record."""
    if stockouts is None:
        return df
    s = stockouts[["Date", "SKU_ID", "Region", "Shipped_Units"]]
    j = df.merge(s, on=["Date", "SKU_ID", "Region"], how="left")
    j.index = df.index
    fix = j["Shipped_Units"].notna() & j["Units_Sold"].ne(j["Shipped_Units"])
    log.add(F, "units_disagree_with_stockout_shipped", fix, "set to Shipped_Units",
            examples=j.loc[fix, ["Date", "SKU_ID", "Region", "Units_Sold", "Shipped_Units"]])
    df.loc[fix, "Units_Sold"] = j.loc[fix, "Shipped_Units"]
    df.loc[fix, "Revenue_USD"] = (df.loc[fix, "Units_Sold"] * df.loc[fix, "Unit_Price_USD"]).round(2)
    _flag(df, fix, "units_from_stockout")
    return df


def check_unique(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    dup = df.duplicated(["Date", "SKU_ID", "Region"], keep=False)
    if dup.any():
        # Same key after label normalisation: keep the latest load (treat as reload).
        log.add(F, "key_collision_after_normalisation", df.duplicated(["Date", "SKU_ID", "Region"]),
                "kept latest Load_Timestamp")
        df = df.sort_values("_ts").drop_duplicates(["Date", "SKU_ID", "Region"], keep="last")
    return df


COLUMNS = [
    "Record_ID", "Date", "Year", "Month", "Product_Family", "Business_Unit", "SKU_ID",
    "SKU_Description", "Customer_Name", "Sales_Channel", "Region", "Units_Sold",
    "Unit_Price_USD", "List_Price_USD", "Revenue_USD", "Load_Timestamp",
    "Units_raw", "Price_raw", "Revenue_raw", "Date_raw", "dq_flags",
]


def clean_train(raw: pd.DataFrame, master: pd.DataFrame, log: DQLog,
                customers: list[str] | None = None,
                stockouts: pd.DataFrame | None = None) -> pd.DataFrame:
    log.count(F, "raw_rows", len(raw))
    df = raw.copy()
    df["dq_flags"] = ""
    df = dedupe(df, log)
    log.count(F, "after_dedupe", len(df))
    df = fix_dates(df, log)
    df = fix_keys(df, master, log, customers)
    df = infer_missing_region(df, log)
    df = fix_price_units_revenue(df, log)
    df = apply_stockout_shipped(df, stockouts, log)
    df = apply_lifecycle(df, master, log)
    df = check_unique(df, log)
    df = df.sort_values(["SKU_ID", "Region", "Date"]).reset_index(drop=True)
    log.count(F, "clean_rows", len(df))
    return df[COLUMNS]
