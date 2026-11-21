"""Invariants on the cleaned outputs (business rules 1-5). Run after `make clean`."""
import json

import numpy as np
import pandas as pd
import pytest

from celeris.config import PROCESSED, REPORTS, TRAIN_END, TRAIN_START

pytestmark = pytest.mark.skipif(not (PROCESSED / "train_clean.parquet").exists(),
                                reason="run the pipeline first")


@pytest.fixture(scope="module")
def train():
    return pd.read_parquet(PROCESSED / "train_clean.parquet")


@pytest.fixture(scope="module")
def master():
    return pd.read_parquet(PROCESSED / "sku_master.parquet")


def test_one_row_per_key(train):
    assert not train.duplicated(["Date", "SKU_ID", "Region"]).any()


def test_dates_in_range_and_year_month_match(train):
    assert train["Date"].between(TRAIN_START, TRAIN_END).all()
    assert (train["Year"] == train["Date"].str[:4].astype(int)).all()
    assert (train["Month"] == train["Date"].str[5:].astype(int)).all()


def test_labels_canonical(train):
    from celeris.config import CHANNELS, REGIONS
    assert set(train["Region"]) <= set(REGIONS)
    assert set(train["Sales_Channel"]) <= set(CHANNELS)
    assert train["SKU_ID"].str.fullmatch(r"[A-Z]{3}-\d{3}").all()
    assert train["Customer_Name"].nunique() <= 15


def test_units_non_negative_integers(train):
    u = train["Units_Sold"].dropna()
    assert (u >= 0).all()
    assert np.allclose(u, u.round())


def test_revenue_reconciles(train):
    t = train.dropna(subset=["Units_Sold"])
    assert np.allclose(t["Revenue_USD"], t["Units_Sold"] * t["Unit_Price_USD"], atol=0.01)


def test_one_price_per_sku_year_across_regions(train):
    # Rule 4: same price in every region; changes only on 1 Jan (+ listed surcharges).
    n = train.groupby(["SKU_ID", "Year"])["List_Price_USD"].nunique()
    assert (n == 1).all()


def test_no_sales_outside_lifecycle(train, master):
    m = master.set_index("SKU_ID")
    eol = train["SKU_ID"].map(m["EOL_Date"])
    launch = train["SKU_ID"].map(m["Launch_Date"])
    assert not (eol.notna() & (train["Date"] > eol)).any()
    assert not (launch.notna() & (train["Date"] < launch)).any()


def test_one_bu_per_family(master):
    assert (master.groupby("Product_Family")["Business_Unit"].nunique() == 1).all()


def test_macro_clean():
    m = pd.read_parquet(PROCESSED / "macro.parquet")
    assert not m["Date"].duplicated().any()
    assert m.drop(columns=["dq_flags"]).notna().all().all()
    assert m["ISM_Manufacturing_PMI"].between(30, 70).all()
    assert len(m) == 113  # 2019-01 .. 2028-05


def test_panel_marks_missing_not_zero():
    p = pd.read_parquet(PROCESSED / "panel.parquet")
    miss = p["row_status"].isin(["missing_row", "after_last_order", "missing_value"])
    assert p.loc[miss, "Units_Sold"].isna().all()
    assert p.loc[~miss, "Units_Sold"].notna().all()


def test_dq_report_machine_readable():
    rep = json.loads((REPORTS / "dq_report.json").read_text())
    assert rep["issues"] and {"file", "issue_type", "rows_affected", "action"} <= set(rep["issues"][0])
