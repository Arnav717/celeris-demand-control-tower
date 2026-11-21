"""Full monthly SKU x Region panel: separates missing rows (unknown) from zeros (no order).

Rule 3: Units_Sold = 0 means no order; a missing row is NOT a zero. The panel makes the
missing months explicit (Units_Sold = NaN, row_status = 'missing_row') so later stages
can impute or mask them instead of silently treating them as zero.
"""
from __future__ import annotations

import pandas as pd

from celeris.clean.dq import DQLog
from celeris.config import REGION_OPEN, TRAIN_END

F = "train.csv"

STATIC = ["Product_Family", "Business_Unit", "SKU_Description"]
SLOW = ["Customer_Name", "Sales_Channel"]


def build_panel(train: pd.DataFrame, master: pd.DataFrame, events_monthly: pd.DataFrame,
                log: DQLog) -> pd.DataFrame:
    m = master.set_index("SKU_ID")
    first = train.groupby(["SKU_ID", "Region"])["Date"].min()
    last = train.groupby(["SKU_ID", "Region"])["Date"].max()

    frames = []
    for (sku, reg), start in first.items():
        end = TRAIN_END
        eol = m.at[sku, "EOL_Date"]
        if isinstance(eol, str) and eol < end:
            end = eol
        start = max(start, REGION_OPEN.get(reg, start))
        months = pd.period_range(start, end, freq="M").astype(str)
        frames.append(pd.DataFrame({"SKU_ID": sku, "Region": reg, "Date": months}))
    grid = pd.concat(frames, ignore_index=True)

    keep = ["Date", "SKU_ID", "Region", "Customer_Name", "Sales_Channel", "Units_Sold",
            "Unit_Price_USD", "Revenue_USD", "dq_flags"]
    p = grid.merge(train[keep], on=["Date", "SKU_ID", "Region"], how="left", indicator=True)
    p["row_status"] = p["_merge"].map({"both": "observed", "left_only": "missing_row"}).astype(str)
    p = p.drop(columns="_merge")
    p["series_last_obs"] = pd.Series(list(zip(p["SKU_ID"], p["Region"]))).map(last).values
    after_last = (p["row_status"] == "missing_row") & (p["Date"] > p["series_last_obs"])
    p.loc[after_last, "row_status"] = "after_last_order"
    p.loc[(p["row_status"] == "observed") & p["Units_Sold"].isna(), "row_status"] = "missing_value"

    # ERP migration in Latin America 2023-03..05 (EV-033): missing rows are a system gap.
    erp = events_monthly[events_monthly["Event_Type"].eq("System") & events_monthly["Scope_Type"].eq("Region")]
    erp_keys = set(zip(erp["Date"], erp["Scope_Value"]))
    is_erp = (p["row_status"] == "missing_row") & pd.Series(list(zip(p["Date"], p["Region"]))).isin(erp_keys).values
    p["missing_reason"] = ""
    p.loc[is_erp, "missing_reason"] = "erp_migration_gap"
    p.loc[(p["row_status"] == "missing_row") & ~is_erp, "missing_reason"] = "unexplained_gap"
    p.loc[after_last, "missing_reason"] = "no_orders_after_last_row"
    mv = p["row_status"] == "missing_value"
    in_erp = pd.Series(list(zip(p["Date"], p["Region"]))).isin(erp_keys).values
    p.loc[mv, "missing_reason"] = "units_unrecoverable"
    p.loc[mv & in_erp, "missing_reason"] = "erp_migration_gap"
    log.add(F, "erp_migration_blank_values", mv & in_erp,
            "kept as NaN; demand imputed from series level (EV-033: transactions not migrated)",
            detail="LatAm 2023-03..05 rows present but Units_Sold and Revenue_USD blank")

    log.add(F, "missing_row_internal", (p["row_status"] == "missing_row"),
            "kept as NaN (not zero, rule 3); imputed for demand history",
            detail=f"of which {int(is_erp.sum())} in LatAm ERP migration window (EV-033)")
    log.add(F, "missing_rows_after_last_order", after_last,
            "kept as NaN; reviewed as possible lost account", category="real_event")

    # Fill slowly-changing attributes from neighbouring rows of the same series.
    p = p.sort_values(["SKU_ID", "Region", "Date"]).reset_index(drop=True)
    g = p.groupby(["SKU_ID", "Region"])
    for c in SLOW:
        p[c] = g[c].transform(lambda s: s.ffill().bfill())
    for c in STATIC:
        p[c] = p["SKU_ID"].map(m[c])
    p["Unit_Price_USD"] = g["Unit_Price_USD"].transform(lambda s: s.ffill().bfill())
    p["dq_flags"] = p["dq_flags"].fillna("")
    p["Year"] = p["Date"].str[:4].astype(int)
    p["Month"] = p["Date"].str[5:].astype(int)
    return p
