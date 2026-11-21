"""Invariants on anomaly / forecast outputs. Run after the full pipeline."""
import numpy as np
import pandas as pd
import pytest

from celeris.config import OUTPUTS, PROCESSED, RAW


def _need(path):
    return pytest.mark.skipif(not path.exists(), reason=f"{path.name} not built")


@_need(PROCESSED / "demand.parquet")
def test_demand_unconstrained_ge_shipments_in_stockouts():
    d = pd.read_parquet(PROCESSED / "demand.parquet")
    s = d[d["stockout_listed"]]
    # shipments may include back-orders carried in from the previous month
    assert (s["demand"] >= s["Units_Sold"] - s["Carried_In_Units"]).all()
    assert (d["demand"].dropna() >= 0).all()
    # imputed months only where the row/value was missing
    assert d.loc[d["demand_imputed"], "row_status"].isin(["missing_row", "missing_value"]).all()


@_need(PROCESSED / "anomaly_register.parquet")
def test_register_has_type_reason_action():
    r = pd.read_parquet(PROCESSED / "anomaly_register.parquet")
    assert r[["anomaly_type", "category", "reason", "action"]].notna().all().all()
    assert {"error", "real_event", "stockout"} <= set(r["category"])


@_need(OUTPUTS / "submission.csv")
def test_submission_matches_test_rows():
    sub = pd.read_csv(OUTPUTS / "submission.csv")
    test = pd.read_csv(RAW / "test.csv") if (RAW / "test.csv").exists() else pd.read_parquet(PROCESSED / "test.parquet")
    assert list(sub.columns) == ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    assert len(sub) == len(test)
    assert (sub[["Date", "SKU_ID", "Region"]].values == test[["Date", "SKU_ID", "Region"]].values).all()
    assert sub["Units_Sold_Forecast"].notna().all()
    assert (sub["Units_Sold_Forecast"] >= 0).all()
    assert np.issubdtype(sub["Units_Sold_Forecast"].dtype, np.integer)


@_need(PROCESSED / "forecast_aggregates.parquet")
def test_aggregates_add_up_and_bands_ordered():
    a = pd.read_parquet(PROCESSED / "forecast_aggregates.parquet")
    fc = pd.read_parquet(PROCESSED / "forecast.parquet")
    assert (a["p10"] <= a["p50"] + 1e-9).all() and (a["p50"] <= a["p90"] + 1e-9).all()
    tot = a[a["level"] == "Total"].set_index("Date")["p50"]
    for lvl in ("Product_Family", "Region"):
        s = a[a["level"] == lvl].groupby("Date")["p50"].sum()
        assert np.allclose(s.reindex(tot.index), tot)
    assert np.isclose(tot.sum(), fc["p50"].sum())


@_need(PROCESSED / "forecast.parquet")
def test_forecast_quantiles_ordered_and_lifecycle_zero():
    fc = pd.read_parquet(PROCESSED / "forecast.parquet")
    assert (fc["p10"] <= fc["p50"] + 1e-9).all() and (fc["p50"] <= fc["p90"] + 1e-9).all()
    m = pd.read_parquet(PROCESSED / "sku_master.parquet").set_index("SKU_ID")
    eol = fc["SKU_ID"].map(m["EOL_Date"])
    after = eol.notna() & (fc["Date"] > eol)
    assert (fc.loc[after, "Units_Sold_Forecast"] == 0).all()
    launch = fc["SKU_ID"].map(m["Launch_Date"])
    assert (fc.loc[fc["Date"] < launch, "Units_Sold_Forecast"] == 0).all()
