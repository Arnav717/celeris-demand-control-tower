"""Cleaning stage: raw CSVs -> standardised, reconciled tables + DQ report."""
from __future__ import annotations

import pandas as pd

from celeris.clean.dq import DQLog
from celeris.clean.labels import norm_region, norm_sku, parse_month
from celeris.clean.panel import build_panel
from celeris.clean.reference import (
    clean_events,
    clean_macro,
    clean_master,
    clean_stockouts,
    clean_test,
)
from celeris.clean.train import clean_train, load_raw
from celeris.config import PROCESSED, RAW, REPORTS


def run(raw_dir=RAW, out_dir=PROCESSED, report_dir=REPORTS) -> dict[str, pd.DataFrame]:
    log = DQLog()
    master = clean_master(pd.read_csv(raw_dir / "sku_master.csv", dtype=str), log)
    test = clean_test(pd.read_csv(raw_dir / "test.csv", dtype=str), master, log)

    so_raw = pd.read_csv(raw_dir / "inventory_stockouts.csv", dtype={"Date": str})
    so_keys = so_raw.assign(Date=so_raw["Date"].map(parse_month), SKU_ID=so_raw["SKU_ID"].map(norm_sku),
                            Region=so_raw["Region"].map(norm_region).fillna(so_raw["Region"]))

    train = clean_train(load_raw(raw_dir / "train.csv"), master, log,
                        customers=sorted(test["Customer_Name"].unique()), stockouts=so_keys)
    stockouts = clean_stockouts(so_raw, train, log)
    macro = clean_macro(pd.read_csv(raw_dir / "macro_indicators.csv", dtype={"Date": str}), log)
    events, events_monthly = clean_events(pd.read_csv(raw_dir / "events_calendar.csv", dtype=str), log)
    panel = build_panel(train, master, events_monthly, log)

    out = {
        "train_clean": train, "panel": panel, "sku_master": master, "test": test,
        "stockouts": stockouts, "macro": macro, "events": events, "events_monthly": events_monthly,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    log.write(report_dir)
    out["dq_log"] = log.to_frame()
    return out
