"""Direct multi-horizon feature matrix.

One row = (series s, forecast origin t, horizon h); target = demand at m = t + h.
Everything derived from demand uses months <= t only (no leakage for any h <= 19).
Exogenous features (calendar, macro, events, lifecycle) are taken at the target month m:
they are known in advance (the macro file and events calendar cover the forecast window).
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from celeris.config import TEST_END, TRAIN_START

MONTHS = pd.period_range(TRAIN_START, TEST_END, freq="M").astype(str).tolist()
MIDX = {m: i for i, m in enumerate(MONTHS)}
MACRO = ["ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
         "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
         "Fuel_Price_Index", "EV_Adoption_Index"]
CATS = ["Product_Family", "Region", "Sales_Channel", "Customer_Name", "Powertrain_Application", "Size_Class"]
INDEX_INDICATORS = {"Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Fuel_Price_Index",
                    "EV_Adoption_Index"}
MACRO_CHG_LAGS = (0, 2, 3, 5)  # leading-indicator lags found in the drivers analysis
EV_LAGS = (0, 3, 6)


# ----------------------------------------------------------------------------- inputs
class Panel:
    """Wide matrices (series x month) + per-series metadata."""

    def __init__(self, demand: pd.DataFrame, test: pd.DataFrame, master: pd.DataFrame,
                 target_col: str = "demand_model"):
        d = demand
        keys = pd.concat([d[["SKU_ID", "Region"]], test[["SKU_ID", "Region"]]]).drop_duplicates()
        keys = keys.sort_values(["SKU_ID", "Region"]).reset_index(drop=True)
        self.keys = keys
        sidx = {k: i for i, k in enumerate(zip(keys["SKU_ID"], keys["Region"]))}
        S, M = len(keys), len(MONTHS)
        self.S, self.M = S, M
        si = np.array([sidx[k] for k in zip(d["SKU_ID"], d["Region"])])
        mi = d["Date"].map(MIDX).to_numpy()

        def wide(col):
            a = np.full((S, M), np.nan)
            a[si, mi] = d[col].to_numpy(dtype=float)
            return a

        self.Y = wide(target_col)          # model target (one-offs capped)
        self.D = wide("demand")            # unconstrained demand (evaluation)
        self.U = wide("Units_Sold")        # constrained shipments
        self.W = np.nan_to_num(wide("sample_weight"), nan=1.0)

        # current attributes: last known (train) or test attributes
        attrs = pd.concat([
            d.sort_values("Date").groupby(["SKU_ID", "Region"])[["Customer_Name", "Sales_Channel"]].last(),
            test.groupby(["SKU_ID", "Region"])[["Customer_Name", "Sales_Channel"]].first(),
        ])
        attrs = attrs[~attrs.index.duplicated(keep="last")]
        meta = keys.join(attrs, on=["SKU_ID", "Region"])
        meta = meta.merge(master[["SKU_ID", "Product_Family", "Powertrain_Application", "Size_Class",
                                  "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"]],
                          on="SKU_ID", how="left")
        self.meta = meta
        self.launch_i = meta["Launch_Date"].map(lambda x: MIDX.get(x, -999 if x and x < MONTHS[0] else 999)).to_numpy()
        self.eol_i = meta["EOL_Date"].map(lambda x: MIDX.get(x, 9999) if isinstance(x, str) else 9999).to_numpy()
        self.first_i = np.argmax(~np.isnan(self.Y), axis=1).astype(float)
        self.first_i[np.isnan(self.Y).all(axis=1)] = np.inf


def _price_rise_by_year(events: pd.DataFrame) -> dict[int, float]:
    out = {}
    for _, r in events[events["Event_Type"].eq("Price Change")].iterrows():
        m = re.search(r"about ([\d.]+)%", r["Description"])
        if m and "Annual" in r["Description"]:
            out[int(r["Start_Month"][:4])] = float(m.group(1))
    return out


class Exog:
    """Month-level and (series, month)-level exogenous features."""

    def __init__(self, panel: Panel, macro: pd.DataFrame, events: pd.DataFrame, em: pd.DataFrame):
        M = panel.M
        mac = macro.set_index("Date").reindex(MONTHS)[MACRO].ffill().bfill().to_numpy()
        self.macro = mac  # (M, 8)
        # index-type indicators in logs (growth), level-type (PMI, %, days) as they are
        self.macro_t = np.column_stack([np.log(mac[:, j]) if n in INDEX_INDICATORS else mac[:, j]
                                        for j, n in enumerate(MACRO)])
        self.rise = _price_rise_by_year(events)
        meta = panel.meta

        def flag(scope_type, scope_col, etypes, share=False):
            e = em[em["Scope_Type"].eq(scope_type) & em["Event_Type"].isin(etypes)]
            val = e.groupby(["Scope_Value", "Date"])["Month_Share" if share else "Event_ID"]
            val = val.sum() if share else val.size().clip(upper=1)
            a = np.zeros((panel.S, M))
            if scope_col is None:
                return a
            look = val.to_dict()
            col = meta[scope_col].to_numpy() if scope_col != "_cust" else \
                (meta["Customer_Name"] + " | " + meta["Region"]).to_numpy()
            for (sv, dt), v in look.items():
                if dt in MIDX:
                    a[col == sv, MIDX[dt]] += v
            return a

        self.holiday = flag("Region", "Region", ["Holiday"], share=True)
        self.disruption = flag("Region", "Region", ["Disruption"])
        self.system = flag("Region", "Region", ["System"])
        self.supply = flag("Product_Family", "Product_Family", ["Supply Constraint"])
        self.logistics = flag("Channel", "Sales_Channel", ["Logistics"])
        cust = em[em["Scope_Type"].eq("Customer")]
        neg = cust[~cust["Description"].str.contains("capacity increase", case=False)]
        pos = cust[cust["Description"].str.contains("capacity increase", case=False)]
        self.cust_neg = self._cust(panel, neg)
        self.cust_pos = self._cust(panel, pos)
        # months since market entry for the region (MEA 2022-01)
        entry = em[em["Event_Type"].eq("Market Entry")].groupby("Scope_Value")["Date"].min()
        self.since_entry = np.full((panel.S, M), 120.0)
        for reg, dt in entry.items():
            rows = meta["Region"].to_numpy() == reg
            self.since_entry[rows] = np.arange(M) - MIDX[dt]

    @staticmethod
    def _cust(panel, ev):
        a = np.zeros((panel.S, panel.M))
        key = (panel.meta["Customer_Name"] + " | " + panel.meta["Region"]).to_numpy()
        for (sv, dt), _ in ev.groupby(["Scope_Value", "Date"]):
            if dt in MIDX:
                a[key == sv, MIDX[dt]] = 1
        return a


# ----------------------------------------------------------------------------- features
def _nanmean(a, axis=1):
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=axis)


def _last_nonzero(Y: np.ndarray) -> np.ndarray:
    """LNZ[s, t] = index of last month <= t with Y > 0 (or -1)."""
    _S, M = Y.shape
    idx = np.where(np.nan_to_num(Y) > 0, np.arange(M), -1)
    return np.maximum.accumulate(idx, axis=1)


def origin_features(P: Panel, t: int) -> dict[str, np.ndarray]:
    """Features of each series at origin t (depend on months <= t only)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _origin_features(P, t)


def _origin_features(P: Panel, t: int) -> dict[str, np.ndarray]:
    Y = P.Y[:, : t + 1]
    f = {}
    for k in (0, 1, 2, 3, 5, 11):
        f[f"lag_{k}"] = Y[:, t - k] if t - k >= 0 else np.full(P.S, np.nan)
    for w in (3, 6, 12, 24, 36, 48):
        f[f"mean_{w}"] = _nanmean(Y[:, max(0, t - w + 1):])
    # robust level: one high or low month moves the median less than the mean
    f["median_12"] = np.nanmedian(Y[:, max(0, t - 11):], axis=1)
    last12 = Y[:, max(0, t - 11):]
    with np.errstate(all="ignore"):
        f["std_12"] = np.nanstd(last12, axis=1)
        f["max_12"] = np.nanmax(last12, axis=1)
        nz = np.where(last12 > 0, last12, np.nan)
        f["nz_mean_12"] = np.nanmean(nz, axis=1)
        f["nz_count_12"] = np.sum(last12 > 0, axis=1).astype(float)
        f["zero_share_24"] = np.nanmean(Y[:, max(0, t - 23):] == 0, axis=1)
        prev12 = _nanmean(Y[:, max(0, t - 23): max(1, t - 11)])
        f["yoy_12"] = f["mean_12"] / (prev12 + 1)
        f["trend_6"] = f["mean_6"] / (_nanmean(Y[:, max(0, t - 11): max(1, t - 5)]) + 1)
        # intermittency: average demand interval and CV^2 of non-zero sizes (last 24m)
        last24 = Y[:, max(0, t - 23):]
        nzc = np.sum(last24 > 0, axis=1)
        obs = np.sum(~np.isnan(last24), axis=1)
        f["adi_24"] = obs / np.maximum(nzc, 1)
        nz24 = np.where(last24 > 0, last24, np.nan)
        f["cv2_24"] = (np.nanstd(nz24, axis=1) / np.nanmean(nz24, axis=1)) ** 2
    f["cycle_est"] = np.clip(np.round(12 / np.maximum(f["nz_count_12"], 1)), 1, 12)
    lnz = _last_nonzero(P.Y[:, : t + 1])[:, -1]
    f["months_since_order"] = np.where(lnz >= 0, t - lnz, 99).astype(float)
    f["history_len"] = (t - P.first_i + 1).clip(0, 999)
    f["_lnz"] = lnz.astype(float)
    f["_scale"] = np.nan_to_num(f["mean_12"], nan=0.0) + 1.0
    return f


def target_features(P: Panel, X: Exog, t: int, h: int, of: dict) -> dict[str, np.ndarray]:
    """Features for target month m = t + h (calendar, seasonal lags, exog)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _target_features(P, X, t, h, of)


def _target_features(P: Panel, X: Exog, t: int, h: int, of: dict) -> dict[str, np.ndarray]:
    m = t + h
    S = P.S
    f = {"h": np.full(S, h, float), "month": np.full(S, int(MONTHS[m][5:]), float)}
    # seasonal lags: same calendar month in the most recent available years (<= t)
    k1 = 12 * int(np.ceil(h / 12))
    for i, k in enumerate((k1, k1 + 12)):
        j = m - k
        f[f"same_month_{i + 1}"] = P.Y[:, j] if 0 <= j <= t else np.full(S, np.nan)
    same = [j for j in range(m % 12, t + 1, 12)]
    Ys = P.Y[:, same] if same else np.full((S, 1), np.nan)
    with np.errstate(all="ignore"):
        f["same_month_mean"] = _nanmean(Ys)
        f["same_month_order_prob"] = np.nanmean(np.where(np.isnan(Ys), np.nan, Ys > 0), axis=1)
        allm = _nanmean(P.Y[:, max(0, t - 47): t + 1])
        f["seasonal_ratio"] = f["same_month_mean"] / (allm + 1e-6)
    cyc = of["cycle_est"]
    lnz = of["_lnz"]
    f["phase_hit"] = np.where(lnz >= 0, ((m - lnz) % cyc) == 0, 0).astype(float)
    # exogenous
    for j, name in enumerate(MACRO):
        for lag in (1, 3, 6):
            f[f"{name}_l{lag}"] = np.full(S, X.macro[m - lag, j])
        f[f"{name}_chg"] = np.full(S, X.macro[m - 1, j] - X.macro[t, j])
        # change between the indicator driving the target month and the one that drove
        # the origin, at each leading lag (macro file covers the forecast window)
        for lag in MACRO_CHG_LAGS:
            f[f"{name}_d{lag}"] = np.full(S, X.macro_t[m - lag, j] - X.macro_t[t - lag, j])
    # EV shift: EV-only parts follow the EV adoption index up, ICE-only parts move the
    # other way, universal parts do not care. The index is known for the forecast window,
    # so give the change from origin to target separately to EV and ICE series (a global
    # macro feature alone made the trees extrapolate EV growth momentum instead).
    ev = X.macro_t[:, MACRO.index("EV_Adoption_Index")]
    pt = P.meta["Powertrain_Application"].to_numpy()
    for lag in EV_LAGS:
        d = ev[m - lag] - ev[t - lag]
        f[f"ev_d{lag}_EV"] = np.where(pt == "EV", d, 0.0)
        f[f"ev_d{lag}_ICE"] = np.where(pt == "ICE", d, 0.0)
    mo, yr = int(MONTHS[m][5:]), int(MONTHS[m][:4])
    f["price_rise_next_jan"] = np.full(S, X.rise.get(yr + 1, 0.0) if mo in (11, 12) else 0.0)
    f["price_rise_this_jan"] = np.full(S, X.rise.get(yr, 0.0) if mo in (1, 2) else 0.0)
    f["holiday_share"] = X.holiday[:, m]
    f["disruption"] = X.disruption[:, m]
    f["system_event"] = X.system[:, m]
    f["supply_constraint"] = X.supply[:, m]
    f["logistics"] = X.logistics[:, m]
    f["cust_neg_event"] = X.cust_neg[:, m]
    f["cust_pos_event"] = X.cust_pos[:, m]
    f["cust_pos_at_origin"] = X.cust_pos[:, t]
    f["months_since_entry"] = X.since_entry[:, m]
    f["months_since_launch"] = (m - P.launch_i).clip(-60, 240).astype(float)
    f["months_to_eol"] = (P.eol_i - m).clip(-60, 120).astype(float)
    return f


def build_frame(P: Panel, X: Exog, origins: list[int], horizons=range(1, 20),
                max_target: int | None = None, require_target: bool = True) -> pd.DataFrame:
    """Stack rows for all series x origins x horizons."""
    parts = []
    for t in origins:
        of = origin_features(P, t)
        enough = of["history_len"] >= 6
        for h in horizons:
            m = t + h
            if m >= P.M or (max_target is not None and m > max_target):
                continue
            tf = target_features(P, X, t, h, of)
            y = P.Y[:, m]
            ok = enough & (P.launch_i <= m) & (P.eol_i >= m)
            if require_target:
                ok &= ~np.isnan(y)
            if not ok.any():
                continue
            cols = {**{k: v[ok] for k, v in of.items()}, **{k: v[ok] for k, v in tf.items()}}
            cols["series"] = np.flatnonzero(ok)
            cols["origin"] = np.full(ok.sum(), t)
            cols["target_m"] = np.full(ok.sum(), m)
            cols["y"] = y[ok]
            cols["y_demand"] = P.D[ok, m]
            cols["w"] = P.W[ok, m]
            parts.append(pd.DataFrame(cols))
    df = pd.concat(parts, ignore_index=True)
    meta = P.meta
    for c in CATS:
        df[c] = pd.Categorical(meta[c].to_numpy()[df["series"]],
                               categories=sorted(meta[c].dropna().unique()))
    num = df.select_dtypes("float64").columns
    df[num] = df[num].astype("float32")
    return df


FEATURES_EXCLUDE = {"series", "origin", "target_m", "y", "y_demand", "w", "_lnz", "_scale"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURES_EXCLUDE]
