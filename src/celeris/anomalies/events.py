"""Match panel rows to the events calendar by month and scope."""
from __future__ import annotations

import pandas as pd


def match_events(rows: pd.DataFrame, events_monthly: pd.DataFrame) -> pd.DataFrame:
    """Return rows' index -> 'EV-xxx;EV-yyy' and event types for events covering that row.

    Scope rules: Global/All; Region; Product_Family; Channel; Customer ('Name | Region').
    """
    em = events_monthly
    out = []
    keys = {
        "Global": lambda r: "All",
        "Region": lambda r: r["Region"],
        "Product_Family": lambda r: r["Product_Family"],
        "Channel": lambda r: r["Sales_Channel"],
        "Customer": lambda r: f"{r['Customer_Name']} | {r['Region']}",
    }
    idx = {(d, st, sv): grp for (d, st, sv), grp in em.groupby(["Date", "Scope_Type", "Scope_Value"])}
    for i, r in rows.iterrows():
        hits = []
        for st, f in keys.items():
            g = idx.get((r["Date"], st, f(r)))
            if g is not None:
                hits.append(g)
        if hits:
            h = pd.concat(hits)
            out.append((i, ";".join(h["Event_ID"]), ";".join(sorted(set(h["Event_Type"]))),
                        " / ".join(h["Description"])))
    res = pd.DataFrame(out, columns=["idx", "event_ids", "event_types", "event_desc"]).set_index("idx")
    return res.reindex(rows.index)
