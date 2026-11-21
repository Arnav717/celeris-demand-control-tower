"""Label / key standardisation helpers (pure functions, unit-tested)."""
from __future__ import annotations

import difflib
import re

import pandas as pd

from celeris.config import CHANNEL_MAP, REGION_MAP

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_SKU_RE = re.compile(r"^([A-Z]{3})[-_ ]?(\d{1,3})$")


def norm_sku(s: str | float) -> str | None:
    """'thm008', ' HYD_006', 'BSS-38' -> 'THM-008', 'HYD-006', 'BSS-038'."""
    if not isinstance(s, str):
        return None
    m = _SKU_RE.match(s.strip().upper())
    return f"{m.group(1)}-{int(m.group(2)):03d}" if m else None


def parse_month(s: str | float) -> str | None:
    """Parse the date formats seen in the ERP extract to 'YYYY-MM'."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if m := re.match(r"^(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?$", s):
        y, mo = int(m[1]), int(m[2])
    elif m := re.match(r"^(\d{1,2})[-/](\d{4})$", s):
        y, mo = int(m[2]), int(m[1])
    elif m := re.match(r"^([A-Za-z]{3})[A-Za-z]*[-\s](\d{4})$", s):
        mo = _MONTHS.get(m[1].lower())
        y = int(m[2])
        if mo is None:
            return None
    else:
        return None
    if not 1 <= mo <= 12:
        return None
    return f"{y:04d}-{mo:02d}"


def norm_region(s: str | float) -> str | None:
    if not isinstance(s, str):
        return None
    return REGION_MAP.get(re.sub(r"\s+", " ", s.strip().lower()))


def norm_channel(s: str | float) -> str | None:
    if not isinstance(s, str):
        return None
    return CHANNEL_MAP.get(re.sub(r"\s+", " ", s.strip().lower()))


_SUFFIX_RE = re.compile(r"[\s,]*\b(inc|ltd|llc|gmbh|corp|co|plc)\.?\s*$", re.I)


def customer_key(name: str) -> str:
    """Letters-only, case-free key without legal suffix: 'APEX TIER ONE Inc.' -> 'apextierone'."""
    return re.sub(r"[^a-z]", "", _SUFFIX_RE.sub("", name.strip()).lower())


def build_customer_map(
    names: pd.Series,
    reference: list[str] | None = None,
    min_count: int = 30,
    cutoff: float = 0.85,
) -> dict[str, str]:
    """Map misspelt customer names to canonical ones.

    Canonical spellings come from `reference` (e.g. the clean keys in test.csv) when
    given; any other frequent key (>= min_count rows) gets its most common spelling.
    Each raw name is matched on `customer_key` (case, spaces, legal suffix), then by
    fuzzy match for typos. Returns raw -> canonical for every raw name.
    """
    clean = names.dropna().map(lambda s: re.sub(r"\s+", " ", s.strip()))
    canon: dict[str, str] = {customer_key(r): r for r in (reference or [])}
    keyed = pd.DataFrame({"name": clean, "key": clean.map(customer_key)})
    key_counts = keyed["key"].value_counts()
    for k, c in key_counts.items():
        if k in canon or c < min_count:
            continue
        if difflib.get_close_matches(k, list(canon), n=1, cutoff=cutoff):
            continue  # a typo of an existing canonical key
        canon[k] = keyed.loc[keyed["key"] == k, "name"].value_counts().index[0]
    out: dict[str, str] = {}
    for raw in names.dropna().unique():
        k = customer_key(raw)
        if k in canon:
            out[raw] = canon[k]
            continue
        match = difflib.get_close_matches(k, list(canon), n=1, cutoff=cutoff)
        out[raw] = canon[match[0]] if match else re.sub(r"\s+", " ", raw.strip())
    return out


def month_add(ym: str, k: int) -> str:
    p = pd.Period(ym, freq="M") + k
    return str(p)


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]
