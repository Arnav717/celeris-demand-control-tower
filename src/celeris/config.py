"""Project paths and domain constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"

TRAIN_START = "2019-01"
TRAIN_END = "2026-10"
TEST_START = "2026-11"
TEST_END = "2028-05"
HORIZON = 19

REGIONS = [
    "North America",
    "Europe",
    "Asia Pacific",
    "India",
    "Latin America",
    "Middle East & Africa",
]
CHANNELS = ["Direct OEM", "Tier-1 Distributor", "Aftermarket", "Export"]

# Lower-cased, whitespace-stripped label -> canonical label.
REGION_MAP = {
    "north america": "North America",
    "n. america": "North America",
    "na": "North America",
    "n america": "North America",
    "europe": "Europe",
    "eu": "Europe",
    "asia pacific": "Asia Pacific",
    "asia-pacific": "Asia Pacific",
    "apac": "Asia Pacific",
    "india": "India",
    "latin america": "Latin America",
    "latin-america": "Latin America",
    "latam": "Latin America",
    "middle east & africa": "Middle East & Africa",
    "middle east and africa": "Middle East & Africa",
    "me&a": "Middle East & Africa",
    "mea": "Middle East & Africa",
}

CHANNEL_MAP = {
    "direct oem": "Direct OEM",
    "direct-oem": "Direct OEM",
    "oem direct": "Direct OEM",
    "tier-1 distributor": "Tier-1 Distributor",
    "tier 1 distributor": "Tier-1 Distributor",
    "t1 distributor": "Tier-1 Distributor",
    "aftermarket": "Aftermarket",
    "after-market": "Aftermarket",
    "export": "Export",
}

# Temporary 6% raw-material surcharge on all invoices (events calendar EV-027).
SURCHARGES = [("2022-07", "2022-12", 1.06)]

# Months during which Celeris operated a region (MEA opened 2022-01, EV-024).
REGION_OPEN = {"Middle East & Africa": "2022-01"}
