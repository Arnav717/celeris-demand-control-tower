"""Dashboard data layer: every view answers, stays JSON-safe, and agrees with the pipeline outputs."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from celeris.config import PROCESSED

pytest.importorskip("fastapi")
if not (PROCESSED / "forecast.parquet").exists():
    pytest.skip("pipeline outputs missing; run `make all` first", allow_module_level=True)

from celeris.dashboard.data import FC_MONTHS, HIST_MONTHS, make_filters, store

SELECTIONS = [
    make_filters(),
    make_filters(family=["Hydraulic Systems"]),
    make_filters(family=["Hydraulic Systems"], region=["Europe"]),
    make_filters(sku=["BSS-001"], region=["Europe"]),
    make_filters(customer=["no such customer"]),
]


@pytest.fixture(scope="module")
def s():
    return store()


def dumps(obj) -> str:
    return json.dumps(obj, allow_nan=False)


@pytest.mark.parametrize("F", SELECTIONS)
def test_views_are_json_safe(s, F):
    for name in ["overview", "health", "explorer", "forecast", "lifecycle", "stockouts"]:
        dumps(getattr(s, name)(F))
    dumps(s.anomalies(F, "", ""))
    dumps(s.whatif(F, "ISM_Manufacturing_PMI", -3.0))
    dumps(s.procurement(F, 6))
    dumps(s.exceptions(F, ""))


def test_meta_and_drivers(s):
    m = s.meta()
    assert m["hist_months"] == HIST_MONTHS and m["fc_months"] == FC_MONTHS
    assert len(m["rows"]) > 3000
    d = s.drivers("", "")
    assert d["family"] in d["families"]


def test_overview_totals_match_forecast_file(s):
    fc = pd.read_parquet(PROCESSED / "forecast.parquet")
    k = s.overview(make_filters())["kpi"]
    assert k["fc19_units"] == pytest.approx(fc["p50"].sum(), rel=1e-9)
    assert k["fc19_rev"] == pytest.approx(fc["Revenue_P50_USD"].sum(), rel=1e-9)
    assert k["fc19_p10"] < k["fc19_units"] < k["fc19_p90"]


def test_filters_partition_the_total(s):
    tot = s.overview(make_filters())["kpi"]["fc19_units"]
    parts = sum(s.overview(make_filters(family=[f]))["kpi"]["fc19_units"] for f in s.meta()["values"]["family"])
    assert parts == pytest.approx(tot, rel=1e-9)


def test_timeline_bands_are_ordered(s):
    for F in SELECTIONS[:4]:
        tl = s.overview(F)["timeline"]
        for lo, mid, hi in zip(tl["p10"], tl["p50"], tl["p90"], strict=True):
            assert lo <= mid + 1e-6 and mid <= hi + 1e-6


def test_zero_shock_changes_nothing(s):
    w = s.whatif(make_filters(), "Raw_Material_Cost_Index", 0.0)
    assert w["scen_total"] == pytest.approx(w["base_total"])


def test_procurement_p90_covers_p50(s):
    p = s.procurement(make_filters(), 6)["kpi"]
    assert p["p90_units"] >= p["p50_units"] > 0


def test_index_versions_static_assets():
    """Asset URLs carry the file mtime so a browser never runs a stale app.js."""
    from celeris.dashboard.server import index

    r = index()
    html = r.body.decode()
    assert r.headers["cache-control"] == "no-cache"
    for asset in ["style.css", "app.js", "vendor/echarts.min.js"]:
        assert f'"/static/{asset}?v=' in html
