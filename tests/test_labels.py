import pandas as pd
import pytest

from celeris.clean.labels import (
    build_customer_map,
    customer_key,
    norm_channel,
    norm_region,
    norm_sku,
    parse_month,
)


@pytest.mark.parametrize("raw,clean", [
    ("THM008", "THM-008"), (" HYD_006", "HYD-006"), ("BSS-38", "BSS-038"),
    ("pmp_015", "PMP-015"), ("PTC-068 ", "PTC-068"), ("XX-1", None), (None, None),
])
def test_norm_sku(raw, clean):
    assert norm_sku(raw) == clean


@pytest.mark.parametrize("raw,clean", [
    ("2019-01", "2019-01"), ("2019-01-01", "2019-01"), ("01/2019", "2019-01"),
    ("2019/03", "2019-03"), ("Jan 2019", "2019-01"), ("Feb-2019", "2019-02"),
    ("12-2020", "2020-12"), ("2019-13", None), ("garbage", None),
])
def test_parse_month(raw, clean):
    assert parse_month(raw) == clean


@pytest.mark.parametrize("raw,clean", [
    ("India ", "India"), ("INDIA", "India"), ("APAC", "Asia Pacific"), ("EU", "Europe"),
    ("LatAm", "Latin America"), ("N. America", "North America"), ("NA", "North America"),
    ("ME&A", "Middle East & Africa"), ("Middle East and Africa", "Middle East & Africa"),
    ("Mars", None),
])
def test_norm_region(raw, clean):
    assert norm_region(raw) == clean


@pytest.mark.parametrize("raw,clean", [
    ("T1 Distributor", "Tier-1 Distributor"), ("OEM Direct", "Direct OEM"),
    ("After-market", "Aftermarket"), ("Export ", "Export"),
])
def test_norm_channel(raw, clean):
    assert norm_channel(raw) == clean


def test_customer_key_strips_case_space_suffix():
    assert customer_key("APEX TIER ONE Inc.") == customer_key("Apex  TierOne") == "apextierone"


def test_customer_map_uses_reference_and_fixes_typos():
    names = pd.Series(["Orion Driveline"] * 50 + ["ORION DRIVELINE"] * 40 + ["Orion Drivline",
                      "Orion Driveline Inc.", "Kestrel Mobility"] + ["Kestrel Mobility"] * 40)
    m = build_customer_map(names, reference=["Orion Driveline"])
    assert m["ORION DRIVELINE"] == "Orion Driveline"
    assert m["Orion Drivline"] == "Orion Driveline"
    assert m["Orion Driveline Inc."] == "Orion Driveline"
    assert m["Kestrel Mobility"] == "Kestrel Mobility"
