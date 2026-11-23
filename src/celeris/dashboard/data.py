"""Dashboard data layer: loads the pipeline outputs once and answers filtered queries.

Everything here is read from `data/processed`, `reports/` and `outputs/`. No number is
typed in by hand: if the pipeline is re-run, the dashboard shows the new results.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from celeris.config import (
    OUTPUTS,
    PROCESSED,
    REPORTS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
)
from celeris.features.build import MACRO
from celeris.features.drivers import (
    EXCLUDE_SHOCKS,
    LEVEL_INDICATORS,
    _demand_growth,
    _indicator_transform,
)

# URL filter key -> column
DIMS = {"family": "Product_Family", "sku": "SKU_ID", "region": "Region",
        "channel": "Sales_Channel", "customer": "Customer_Name"}
SERIES_KEYS = ["SKU_ID", "Region", "Customer_Name", "Sales_Channel"]
FINAL_MODEL = "gbm_tweedie"
BASELINES = ["seasonal_naive", "moving_average_12", "seasonal_index_x_level", "tsb"]
MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive", "moving_average_12": "Moving average (12m)",
    "seasonal_index_x_level": "Seasonal index x level", "tsb": "TSB (intermittent)",
    "gbm_tweedie": "LightGBM Tweedie (final)", "gbm_l1": "LightGBM L1", "ensemble": "Ensemble",
}


def months(a: str, b: str) -> list[str]:
    return [str(p) for p in pd.period_range(a, b, freq="M")]


HIST_MONTHS = months(TRAIN_START, TRAIN_END)
FC_MONTHS = months(TEST_START, TEST_END)
LAST12 = HIST_MONTHS[-12:]
NEXT12 = FC_MONTHS[:12]

Filters = tuple[tuple[str, tuple[str, ...]], ...]


def make_filters(**kw: list[str] | None) -> Filters:
    return tuple((k, tuple(sorted(set(v)))) for k, v in sorted(kw.items()) if v)


def clean(o):
    """Make an object JSON-safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if o is pd.NA or o is pd.NaT:
        return None
    return o


def records(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict("records"))


def wape(y: pd.Series, p: pd.Series) -> float | None:
    s = float(y.sum())
    return float((y - p).abs().sum() / s) if s > 0 else None


def bias(y: pd.Series, p: pd.Series) -> float | None:
    s = float(y.sum())
    return float((p.sum() - s) / s) if s > 0 else None


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


class Store:
    def __init__(self) -> None:
        d = _read("demand")
        d["unserved"] = (d["demand"] - d["Units_Sold"].fillna(0)).clip(lower=0).where(
            d["stockout_listed"] | d["stockout_inferred"], 0.0)
        self.hist = d[["Date", *SERIES_KEYS, "Product_Family", "Units_Sold", "demand", "Revenue_USD",
                       "Unit_Price_USD", "lost_units_est", "Backordered_Units", "Reason", "stockout_listed",
                       "stockout_inferred", "demand_source", "row_status", "one_off_order", "unserved"]].copy()

        fc = _read("forecast")
        fc["lo2"] = (fc["p50"] - fc["p10"]).clip(lower=0) ** 2
        fc["hi2"] = (fc["p90"] - fc["p50"]).clip(lower=0) ** 2
        fc["series"] = fc[SERIES_KEYS].astype(str).agg("|".join, axis=1)
        self.fc = fc
        self.fa = _read("forecast_aggregates")
        self.factors = _read("aggregate_interval_factors")

        self.bt = _read("backtest_predictions")
        self.anom = _read("anomaly_register")
        self.master = _read("sku_master")
        self.profile = _read("series_profile")
        self.events = _read("events")
        self.events_m = _read("events_monthly")
        self.family_breaks = _read("family_breaks")
        self.macro = _read("macro")
        self.best_lag = _read("drivers_best_lag")
        self.leadlag = _read("drivers_leadlag")
        self.stability = _read("drivers_stability")
        self.ev = _read("ev_shift")
        self.train = _read("train_clean")[["Date", "Product_Family", *SERIES_KEYS, "Units_raw", "Units_Sold",
                                           "Revenue_raw", "Revenue_USD", "dq_flags"]]
        self.dq = json.loads((REPORTS / "dq_report.json").read_text())
        self.anom_summary = json.loads((REPORTS / "anomaly_summary.json").read_text())
        self.importance = json.loads((REPORTS / "feature_importance.json").read_text())

        dims = pd.concat([self.hist[list(DIMS.values())], fc[list(DIMS.values())]]).drop_duplicates()
        self.dims = dims.sort_values(list(DIMS.values())).reset_index(drop=True)
        self._betas = self._fit_betas()
        self._series = self._series_table()

    # ---------------------------------------------------------------- helpers
    def mask(self, df: pd.DataFrame, F: Filters) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        for k, vals in F:
            col = DIMS[k]
            if col in df.columns:
                m &= df[col].isin(vals).to_numpy()
        return m

    def sel(self, df: pd.DataFrame, F: Filters) -> pd.DataFrame:
        return df[self.mask(df, F)] if F else df

    def skus(self, F: Filters) -> list[str]:
        return sorted(self.sel(self.dims, F)["SKU_ID"].unique())

    def meta(self) -> dict:
        cols = list(DIMS.values())
        values = {k: sorted(self.dims[c].dropna().unique()) for k, c in DIMS.items()}
        idx = {k: {v: i for i, v in enumerate(values[k])} for k in DIMS}
        rows = [[idx[k][r[c]] for k, c in DIMS.items()] for r in self.dims[cols].to_dict("records")]
        sku_desc = dict(zip(self.master["SKU_ID"], self.master["SKU_Description"]))
        return clean({"dims": list(DIMS), "values": values, "rows": rows, "sku_desc": sku_desc,
                      "train_end": TRAIN_END, "test_start": TEST_START, "test_end": TEST_END,
                      "hist_months": HIST_MONTHS, "fc_months": FC_MONTHS})

    # ---------------------------------------------------------------- forecast bands
    def _agg_member(self, F: Filters) -> tuple[str, str] | None:
        if not F:
            return ("Total", "All")
        if len(F) == 1 and len(F[0][1]) == 1:
            k, (v,) = F[0]
            if k == "family":
                return ("Product_Family", v)
            if k == "region":
                return ("Region", v)
        return None

    def fc_monthly(self, F: Filters, fc: pd.DataFrame | None = None) -> pd.DataFrame:
        """Monthly P10/P50/P90 for the selection.

        Whole-company, single family or single region: the calibrated aggregate bands from
        the pipeline. One series: its own quantiles. Any other subset: series half-widths
        combined in quadrature (independent errors), floored at the backtest spread of a
        family-level aggregate, so the band is neither the too-wide plain sum nor too narrow.
        """
        member = self._agg_member(F) if fc is None else None
        fc = self.sel(self.fc, F) if fc is None else fc
        g = fc.groupby("Date").agg(p50=("p50", "sum"), rev=("Revenue_P50_USD", "sum"), lo2=("lo2", "sum"),
                                   hi2=("hi2", "sum"), p10s=("p10", "sum"), p90s=("p90", "sum"))
        g = g.reindex(FC_MONTHS).fillna(0.0)
        n_series = fc["series"].nunique()
        if member is not None:
            a = self.fa[(self.fa["level"] == member[0]) & (self.fa["member"] == member[1])].set_index("Date")
            ratio_lo = (a["p10"] / a["p50"]).reindex(FC_MONTHS).fillna(1.0)
            ratio_hi = (a["p90"] / a["p50"]).reindex(FC_MONTHS).fillna(1.0)
            g["p10"], g["p90"] = g["p50"] * ratio_lo, g["p50"] * ratio_hi
            g["band"] = f"calibrated {member[0].replace('_', ' ').lower()} band"
        elif n_series <= 1:
            g["p10"], g["p90"] = g["p10s"], g["p90s"]
            g["band"] = "series quantiles"
        else:
            f = self.factors[self.factors["level"] == "Product_Family"].set_index("h_bucket")
            hb = pd.cut(pd.Series(range(1, len(FC_MONTHS) + 1)), [0, 6, 12, 19], labels=["1-6", "7-12", "13-19"])
            lo_f = (f["ratio_p10"] / f["ratio_p50"]).reindex(hb.astype(str)).to_numpy()
            hi_f = (f["ratio_p90"] / f["ratio_p50"]).reindex(hb.astype(str)).to_numpy()
            p50 = g["p50"].to_numpy()
            g["p10"] = p50 - np.maximum(np.sqrt(g["lo2"].to_numpy()), p50 * np.clip(1 - lo_f, 0, None))
            g["p90"] = p50 + np.maximum(np.sqrt(g["hi2"].to_numpy()), p50 * np.clip(hi_f - 1, 0, None))
            g["p10"] = g["p10"].clip(lower=0)
            g["band"] = "combined series band"
        ratio = np.where(g["p50"] > 0, 1.0, np.nan)
        g["rev_p10"] = g["rev"] * np.where(g["p50"] > 0, g["p10"] / g["p50"].where(g["p50"] > 0), ratio)
        g["rev_p90"] = g["rev"] * np.where(g["p50"] > 0, g["p90"] / g["p50"].where(g["p50"] > 0), ratio)
        return g

    def hist_monthly(self, F: Filters) -> pd.DataFrame:
        h = self.sel(self.hist, F)
        g = h.groupby("Date").agg(demand=("demand", "sum"), shipped=("Units_Sold", "sum"),
                                  revenue=("Revenue_USD", "sum"), unserved=("unserved", "sum"))
        return g.reindex(HIST_MONTHS)

    def timeline(self, F: Filters) -> dict:
        h, f = self.hist_monthly(F), self.fc_monthly(F)
        return {"hist_months": HIST_MONTHS, "fc_months": FC_MONTHS,
                "demand": h["demand"].to_numpy(), "shipped": h["shipped"].to_numpy(),
                "revenue": h["revenue"].to_numpy(),
                "p10": f["p10"].to_numpy(), "p50": f["p50"].to_numpy(), "p90": f["p90"].to_numpy(),
                "rev_p10": f["rev_p10"].to_numpy(), "rev_p50": f["rev"].to_numpy(),
                "rev_p90": f["rev_p90"].to_numpy(), "band": str(f["band"].iloc[0]) if len(f) else ""}

    # ---------------------------------------------------------------- 1. executive overview
    @lru_cache(maxsize=256)  # noqa: B019 (one Store per process)
    def overview(self, F: Filters) -> dict:
        h, fcs = self.sel(self.hist, F), self.sel(self.fc, F)
        f = self.fc_monthly(F)
        hm = self.hist_monthly(F)
        prior19 = float(hm["demand"].iloc[-19:].sum())
        last12, next12 = float(hm["demand"].iloc[-12:].sum()), float(f["p50"].iloc[:12].sum())
        rev_last12 = float(hm["revenue"].iloc[-12:].sum())
        mix = {}
        for key, col in [("family", "Product_Family"), ("region", "Region"), ("channel", "Sales_Channel")]:
            a = h[h["Date"].isin(LAST12)].groupby(col).agg(hist12=("demand", "sum"), hist12_rev=("Revenue_USD", "sum"))
            b = fcs.groupby(col).agg(fc19=("p50", "sum"), fc19_rev=("Revenue_P50_USD", "sum"))
            c = fcs[fcs["Date"].isin(NEXT12)].groupby(col).agg(fc12=("p50", "sum"), fc12_rev=("Revenue_P50_USD", "sum"))
            m = a.join(b, how="outer").join(c, how="outer").fillna(0).reset_index().rename(columns={col: "name"})
            m["growth"] = np.where(m["hist12"] > 0, m["fc12"] / m["hist12"] - 1, np.nan)
            mix[key] = records(m.sort_values("fc19", ascending=False))
        acc = self._accuracy_kpis(F)
        new = fcs[fcs["source"] == "new_launch_rule"]["SKU_ID"].nunique()
        eol = fcs[fcs["lifecycle_rule"].isin(["eol_rundown", "after_eol_zero"])]["SKU_ID"].nunique()
        kpi = {
            "fc19_units": float(f["p50"].sum()), "fc19_p10": float(f["p10"].sum()), "fc19_p90": float(f["p90"].sum()),
            "fc19_rev": float(f["rev"].sum()), "prior19_units": prior19,
            "growth19": (float(f["p50"].sum()) / prior19 - 1) if prior19 else None,
            "next12_units": next12, "last12_units": last12,
            "growth12": (next12 / last12 - 1) if last12 else None,
            "next12_rev": float(f["rev"].iloc[:12].sum()), "last12_rev": rev_last12,
            "series": int(fcs["series"].nunique()), "skus": int(fcs["SKU_ID"].nunique()),
            "new_launch_skus": int(new), "eol_skus": int(eol), **acc,
        }
        return clean({"timeline": self.timeline(F), "mix": mix, "kpi": kpi})

    def _accuracy_kpis(self, F: Filters) -> dict:
        b = self.sel(self.bt, F)
        if b.empty:
            return {"wape": None, "agg_wape": None, "bias": None, "coverage": None}
        a = b.groupby(["cutoff", "Date"])[["y_demand", FINAL_MODEL]].sum()
        lo = np.minimum(b["q10"], b[FINAL_MODEL])
        hi = np.maximum(b["q90"], b[FINAL_MODEL])
        cov = float(((b["y_demand"] >= lo) & (b["y_demand"] <= hi)).mean())
        return {"wape": wape(b["y_demand"], b[FINAL_MODEL]), "agg_wape": wape(a["y_demand"], a[FINAL_MODEL]),
                "bias": bias(b["y_demand"], b[FINAL_MODEL]), "coverage": cov,
                "naive_wape": wape(b["y_demand"], b["seasonal_naive"])}

    # ---------------------------------------------------------------- 2. data health
    @lru_cache(maxsize=128)  # noqa: B019 (one Store per process)
    def health(self, F: Filters) -> dict:
        t = self.sel(self.train, F)
        g = t.groupby("Date").agg(raw=("Units_raw", "sum"), clean=("Units_Sold", "sum"),
                                  raw_rev=("Revenue_raw", "sum"), clean_rev=("Revenue_USD", "sum"))
        g = g.reindex(HIST_MONTHS)
        dem = self.hist_monthly(F)["demand"]
        flags = t["dq_flags"].fillna("").str.split(";").explode()
        flags = flags[flags.str.len() > 0].value_counts()
        a = self.sel(self.anom, F)
        err = a[a["category"] == "error"]["anomaly_type"].value_counts()
        h = self.sel(self.hist, F)
        status = h["row_status"].value_counts()
        dsrc = h["demand_source"].replace("", "other").value_counts()
        issues = pd.read_csv(REPORTS / "dq_issues.csv").drop(columns=["examples"]).fillna("")
        rc = [{"file": k, **v} for k, v in self.dq["row_counts"].items()]
        raw_neg = int((t["Units_raw"] < 0).sum())
        return clean({
            "months": HIST_MONTHS, "raw_units": g["raw"].to_numpy(), "clean_units": g["clean"].to_numpy(),
            "demand": dem.to_numpy(), "raw_rev": g["raw_rev"].to_numpy(), "clean_rev": g["clean_rev"].to_numpy(),
            "row_flags": [{"flag": k, "rows": v} for k, v in flags.items()],
            "error_types": [{"type": k, "rows": v} for k, v in err.items()],
            "row_status": [{"status": k, "rows": v} for k, v in status.items()],
            "demand_source": [{"source": k, "rows": v} for k, v in dsrc.items()],
            "issues": records(issues), "row_counts": rc, "n_issue_types": self.dq.get("n_issue_types"),
            "kpi": {"rows": len(t), "rows_flagged": int((t["dq_flags"].fillna("") != "").sum()),
                    "raw_negative": raw_neg, "missing_months": int(status.get("missing_row", 0)),
                    "raw_total": float(t["Units_raw"].sum()), "clean_total": float(t["Units_Sold"].sum()),
                    "demand_total": float(dem.sum())},
        })

    # ---------------------------------------------------------------- 3. demand explorer
    @lru_cache(maxsize=256)  # noqa: B019 (one Store per process)
    def explorer(self, F: Filters) -> dict:
        h = self.sel(self.hist, F)
        keys = dict(F)
        if "family" not in keys:
            level = "family"
        elif "sku" not in keys:
            level = "sku"
        elif "region" not in keys:
            level = "region"
        else:
            level = "customer"
        col = DIMS[level]
        cur = h[h["Date"].isin(LAST12)].groupby(col)["demand"].sum()
        prev = h[h["Date"].isin(HIST_MONTHS[-24:-12])].groupby(col)["demand"].sum()
        zero = h.assign(z=h["demand"].fillna(0) <= 0).groupby(col)["z"].mean()
        br = pd.DataFrame({"last12": cur, "prev12": prev, "zero_share": zero}).fillna({"last12": 0, "prev12": 0})
        br["yoy"] = np.where(br["prev12"] > 0, br["last12"] / br["prev12"] - 1, np.nan)
        br = br.reset_index().rename(columns={col: "name"}).sort_values("last12", ascending=False)
        if level == "sku":
            br["desc"] = br["name"].map(dict(zip(self.master["SKU_ID"], self.master["SKU_Description"])))
        hm = self.hist_monthly(F)
        dem = hm["demand"]
        ma12 = dem.rolling(12, min_periods=12).mean()
        # seasonality: demand by year x calendar month, and the average seasonal index
        s = dem.to_frame("v").assign(year=[m[:4] for m in HIST_MONTHS], mon=[int(m[5:]) for m in HIST_MONTHS])
        yr_mean = s.groupby("year")["v"].transform("mean")
        s["idx"] = s["v"] / yr_mean
        heat = [[int(r.mon) - 1, int(r.year) - int(HIST_MONTHS[0][:4]), r.v] for r in s.itertuples()]
        sidx = s[~s["year"].isin(["2020", "2021"])].groupby("mon")["idx"].mean()
        # zero months: share of series-months with zero demand, by month
        zm = h.assign(z=h["demand"].fillna(0) <= 0).groupby("Date")["z"].mean().reindex(HIST_MONTHS)
        prof = self.profile.merge(self.sel(self.dims, F)[["SKU_ID", "Region"]].drop_duplicates(), on=["SKU_ID", "Region"])
        icls = prof["intermittency_class"].value_counts()
        ms = self.master[self.master["SKU_ID"].isin(self.skus(F))]
        life = ms["Lifecycle_Status"].value_counts()
        launches = ms.assign(y=ms["Launch_Date"].str[:4]).groupby("y").size()
        lifecycle_rows = ms[(ms["Lifecycle_Status"] != "Active") | ms["Replaces_SKU"].notna()]
        return clean({
            "level": level, "breakdown": records(br.head(40)), "n_items": len(br),
            "months": HIST_MONTHS, "demand": dem.to_numpy(), "shipped": hm["shipped"].to_numpy(), "ma12": ma12.to_numpy(),
            "heat": heat, "years": sorted(s["year"].unique()), "season_idx": [sidx.get(m) for m in range(1, 13)],
            "zero_share_month": zm.to_numpy(), "intermittency": [{"cls": k, "n": v} for k, v in icls.items()],
            "lifecycle_status": [{"status": k, "n": v} for k, v in life.items()],
            "launch_years": [{"year": k, "n": v} for k, v in launches.items()],
            "lifecycle_rows": records(lifecycle_rows[["SKU_ID", "SKU_Description", "Powertrain_Application",
                                                       "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"]]),
            "kpi": {"series": len(prof), "skus": len(ms), "zero_share": float((h["demand"].fillna(0) <= 0).mean()) if len(h) else None,
                    "last12": float(dem.iloc[-12:].sum()), "prev12": float(dem.iloc[-24:-12].sum())},
        })

    # ---------------------------------------------------------------- 4. anomaly monitor
    @lru_cache(maxsize=256)  # noqa: B019 (one Store per process)
    def anomalies(self, F: Filters, category: str = "", atype: str = "") -> dict:
        a = self.sel(self.anom, F)
        hm = self.hist_monthly(F)
        cats = a["category"].value_counts()
        types = a.groupby(["category", "anomaly_type"]).size().reset_index(name="n").sort_values("n", ascending=False)
        pts = a[a["scope"] == "point"]
        marks = pts.groupby(["Date", "category"]).agg(n=("anomaly_type", "size"),
                                                     top=("anomaly_type", lambda s: s.value_counts().index[0])).reset_index()
        series_ev = a[a["scope"] != "point"][["Date", "SKU_ID", "Region", "Product_Family", "scope", "anomaly_type",
                                               "reason", "action"]]
        fams = sorted(set(self.sel(self.dims, F)["Product_Family"]))
        breaks = self.family_breaks[self.family_breaks["Product_Family"].isin(fams)]
        tab = a
        if category:
            tab = tab[tab["category"] == category]
        if atype:
            tab = tab[tab["anomaly_type"] == atype]
        tab = tab.sort_values(["Date", "SKU_ID"], ascending=[False, True])
        regions = set(self.sel(self.dims, F)["Region"])
        ev = self.events[(self.events["Scope_Type"] == "Global")
                         | ((self.events["Scope_Type"] == "Region") & self.events["Scope_Value"].isin(regions))
                         | ((self.events["Scope_Type"] == "Product_Family") & self.events["Scope_Value"].isin(fams))
                         | (~self.events["Scope_Type"].isin(["Global", "Region", "Product_Family"]))]
        return clean({
            "months": HIST_MONTHS, "demand": hm["demand"].to_numpy(), "shipped": hm["shipped"].to_numpy(),
            "categories": [{"category": k, "n": v} for k, v in cats.items()], "types": records(types),
            "marks": records(marks), "series_events": records(series_ev.sort_values("Date").tail(300)),
            "breaks": records(breaks), "table": records(tab.head(500)), "table_total": len(tab),
            "events": records(ev[["Event_ID", "Start_Month", "End_Month", "Scope_Type", "Scope_Value", "Event_Type",
                                  "Description"]]),
        })

    # ---------------------------------------------------------------- 5. forecast & accuracy
    @lru_cache(maxsize=256)  # noqa: B019 (one Store per process)
    def forecast(self, F: Filters) -> dict:
        b = self.sel(self.bt, F)
        models = [*BASELINES, FINAL_MODEL, "gbm_l1", "ensemble"]
        board = []
        agg = b.groupby(["cutoff", "Date"])[["y_demand", *models]].sum()
        for m in models:
            board.append({"model": m, "label": MODEL_LABELS[m], "wape": wape(b["y_demand"], b[m]),
                          "bias": bias(b["y_demand"], b[m]), "agg_wape": wape(agg["y_demand"], agg[m]),
                          "final": m == FINAL_MODEL})

        def by(col: str) -> list[dict]:
            out = []
            for k, g in b.groupby(col):
                out.append({"name": k, "wape": wape(g["y_demand"], g[FINAL_MODEL]),
                            "naive_wape": wape(g["y_demand"], g["seasonal_naive"]),
                            "bias": bias(g["y_demand"], g[FINAL_MODEL]), "actual": float(g["y_demand"].sum())})
            return out

        fan = []
        for c, g in agg.reset_index().groupby("cutoff"):
            fan.append({"cutoff": c, "dates": g["Date"].tolist(), "pred": g[FINAL_MODEL].to_numpy(),
                        "actual": g["y_demand"].to_numpy()})
        lo = np.minimum(b["q10"], b[FINAL_MODEL])
        hi = np.maximum(b["q90"], b[FINAL_MODEL])
        inside = (b["y_demand"] >= lo) & (b["y_demand"] <= hi)
        cov = []
        for c, g in b.assign(inside=inside, below=b["y_demand"] < lo, above=b["y_demand"] > hi).groupby("cutoff"):
            cov.append({"cutoff": c, "coverage": float(g["inside"].mean()), "below": float(g["below"].mean()),
                        "above": float(g["above"].mean())})
        imp = self.importance.get(FINAL_MODEL, [])[:15]
        fcs = self.sel(self.fc, F)
        src = fcs.groupby("source")["p50"].sum()
        return clean({"timeline": self.timeline(F), "board": board, "by_family": by("Product_Family"),
                      "by_region": by("Region"), "by_channel": by("Sales_Channel"), "by_horizon": by("h"),
                      "fan": fan, "coverage": cov, "importance": imp, "kpi": self._accuracy_kpis(F),
                      "sources": [{"source": k, "units": v} for k, v in src.items()]})

    # ---------------------------------------------------------------- 6. drivers (macro lead / lag)
    @lru_cache(maxsize=32)  # noqa: B019 (one Store per process)
    def drivers(self, family: str, indicator: str) -> dict:
        bl = self.best_lag
        fams = sorted(bl["Product_Family"].unique())
        family = family if family in fams else fams[0]
        indicator = indicator if indicator in MACRO else bl[bl["Product_Family"] == family].iloc[0]["indicator"]
        mat = [[MACRO.index(r.indicator), fams.index(r.Product_Family), r.corr, r.lag] for r in bl.itertuples()]
        ll = self.leadlag[self.leadlag["Product_Family"] == family]
        llm = [[int(r.lag), MACRO.index(r.indicator), r.corr] for r in ll.itertuples()]
        stab = self.stability[self.stability["Product_Family"] == family].sort_values(["indicator", "window"])
        growth = _demand_growth(self.hist, "Product_Family")
        ind = _indicator_transform(self.macro)
        lag = int(bl[(bl["Product_Family"] == family) & (bl["indicator"] == indicator)]["lag"].iloc[0])
        return clean({
            "families": fams, "indicators": MACRO, "family": family, "indicator": indicator, "lag": lag,
            "best_matrix": mat, "leadlag": llm, "stability": records(stab),
            "months": growth.index.tolist(), "growth": growth[family].to_numpy(),
            "ind_months": ind.index.tolist(), "ind": ind[indicator].shift(lag).to_numpy(),
            "ind_raw": self.macro.set_index("Date")[indicator].reindex(ind.index).to_numpy(),
            "transform": "level (3m avg)" if indicator in LEVEL_INDICATORS else "YoY log change (3m avg)",
            "ev": records(self.ev), "top": records(bl.sort_values("abs_corr", ascending=False)
                                                   .drop_duplicates("Product_Family")),
        })

    # ---------------------------------------------------------------- 7. what-if
    def _fit_betas(self) -> pd.DataFrame:
        g = _demand_growth(self.hist, "Product_Family")
        ind = _indicator_transform(self.macro)
        rows = []
        for r in self.best_lag.itertuples():
            y = g[r.Product_Family]
            x = ind[r.indicator].shift(r.lag).reindex(y.index)
            m = y.notna() & x.notna() & ~y.index.to_series().between(*EXCLUDE_SHOCKS)
            if m.sum() < 18:
                continue
            beta, _ = np.polyfit(x[m], y[m], 1)
            rows.append({"Product_Family": r.Product_Family, "indicator": r.indicator, "lag": int(r.lag),
                         "beta": float(beta), "corr": float(r.corr), "r2": float(r.corr ** 2), "n": int(m.sum())})
        return pd.DataFrame(rows)

    @lru_cache(maxsize=256)  # noqa: B019 (one Store per process)
    def whatif(self, F: Filters, indicator: str, shock: float) -> dict:
        """Sustained shock to one indicator from the first forecast month.

        Effect per family from the historical regression of demand YoY growth on the
        indicator at its best lead (COVID excluded). Level indicators (PMI, utilisation...)
        shift YoY growth for as long as the shock lasts, so the effect compounds after 12
        months; growth-type indices (cost, orders, fuel...) are shocked in % of the index and
        move the demand level once.
        """
        fcs = self.sel(self.fc, F)
        bt = self._betas[self._betas["indicator"] == indicator].set_index("Product_Family")
        level = indicator in LEVEL_INDICATORS
        dx = shock if level else math.log1p(shock / 100.0)
        t = np.arange(1, len(FC_MONTHS) + 1)
        mult = {}
        fam_rows = []
        for fam, g in fcs.groupby("Product_Family"):
            if fam not in bt.index:
                mult[fam] = np.ones(len(t))
                continue
            r = bt.loc[fam]
            on = t > r["lag"]
            k = np.where(on, ((t - r["lag"] - 1) // 12 + 1) if level else 1, 0)
            mult[fam] = np.exp(r["beta"] * dx * k)
            base = g.groupby("Date")["p50"].sum().reindex(FC_MONTHS).fillna(0).to_numpy()
            rev = g.groupby("Date")["Revenue_P50_USD"].sum().reindex(FC_MONTHS).fillna(0).to_numpy()
            fam_rows.append({"family": fam, "lag": int(r["lag"]), "beta": r["beta"], "corr": r["corr"], "r2": r["r2"],
                             "base": float(base.sum()), "scen": float((base * mult[fam]).sum()),
                             "base_rev": float(rev.sum()), "scen_rev": float((rev * mult[fam]).sum())})
        mi = {m: i for i, m in enumerate(FC_MONTHS)}
        idx = fcs["Date"].map(mi).to_numpy()
        fm = np.array([mult.get(f, np.ones(len(t)))[i] for f, i in zip(fcs["Product_Family"], idx)]) if len(fcs) else np.array([])
        scen = fcs.assign(p50=fcs["p50"] * fm, Revenue_P50_USD=fcs["Revenue_P50_USD"] * fm)
        base_m = fcs.groupby("Date")["p50"].sum().reindex(FC_MONTHS).fillna(0)
        scen_m = scen.groupby("Date")["p50"].sum().reindex(FC_MONTHS).fillna(0)
        hm = self.hist_monthly(F)
        return clean({"indicator": indicator, "shock": shock, "level_type": level,
                      "unit": "points" if level else "% of index", "hist_months": HIST_MONTHS[-24:],
                      "hist": hm["demand"].iloc[-24:].to_numpy(), "months": FC_MONTHS,
                      "base": base_m.to_numpy(), "scen": scen_m.to_numpy(), "families": fam_rows,
                      "base_total": float(base_m.sum()), "scen_total": float(scen_m.sum()),
                      "base_rev": float(fcs["Revenue_P50_USD"].sum()), "scen_rev": float(scen["Revenue_P50_USD"].sum()),
                      "indicators": MACRO, "current": self.macro.set_index("Date")[indicator].iloc[-1]})

    # ---------------------------------------------------------------- 8. procurement (P90)
    def procurement_table(self, F: Filters, window: int) -> pd.DataFrame:
        fcs = self.sel(self.fc, F)
        fcs = fcs[fcs["Date"].isin(FC_MONTHS[:window])]
        g = fcs.assign(p90_value=fcs["p90"] * fcs["Unit_Price_USD"], p50_value=fcs["p50"] * fcs["Unit_Price_USD"])
        t = g.groupby(["SKU_ID", "Product_Family"]).agg(
            regions=("Region", "nunique"), p50=("p50", "sum"), p90=("p90", "sum"),
            p50_value=("p50_value", "sum"), p90_value=("p90_value", "sum")).reset_index()
        t["buffer"] = t["p90"] - t["p50"]
        t["buffer_value"] = t["p90_value"] - t["p50_value"]
        t["buffer_pct"] = np.where(t["p50"] > 0, t["buffer"] / t["p50"], np.nan)
        t["unit_cost_proxy"] = np.where(t["p90"] > 0, t["p90_value"] / t["p90"], np.nan)
        t["SKU_Description"] = t["SKU_ID"].map(dict(zip(self.master["SKU_ID"], self.master["SKU_Description"])))
        t["Lifecycle_Status"] = t["SKU_ID"].map(dict(zip(self.master["SKU_ID"], self.master["Lifecycle_Status"])))
        for c in ["p50", "p90", "buffer"]:
            t[c] = np.ceil(t[c])
        return t.sort_values("p90_value", ascending=False)

    @lru_cache(maxsize=256)  # noqa: B019 (one Store per process)
    def procurement(self, F: Filters, window: int) -> dict:
        t = self.procurement_table(F, window)
        fcs = self.sel(self.fc, F)
        m = fcs.groupby("Date").agg(p50=("p50", "sum"), p90=("p90", "sum"), p10=("p10", "sum")).reindex(FC_MONTHS).fillna(0)
        fam = t.groupby("Product_Family")[["p50_value", "p90_value", "buffer_value"]].sum().reset_index()
        return clean({"window": window, "months": FC_MONTHS, "p50": m["p50"].to_numpy(), "p90": m["p90"].to_numpy(),
                      "table": records(t.head(300)), "n": len(t), "by_family": records(fam),
                      "kpi": {"p50_units": float(t["p50"].sum()), "p90_units": float(t["p90"].sum()),
                              "p50_value": float(t["p50_value"].sum()), "p90_value": float(t["p90_value"].sum()),
                              "buffer_value": float(t["buffer_value"].sum()), "skus": len(t)}})

    # ---------------------------------------------------------------- 9. lifecycle
    @lru_cache(maxsize=128)  # noqa: B019 (one Store per process)
    def lifecycle(self, F: Filters) -> dict:
        ms = self.master[self.master["SKU_ID"].isin(self.skus(F))]
        fcs = self.sel(self.fc, F)
        h = self.sel(self.hist, F)

        def group(r):
            if r["source"] == "new_launch_rule":
                return "New launch"
            if r["lifecycle_rule"] in ("eol_rundown", "after_eol_zero"):
                return "EOL run-down"
            return "Active"

        fg = fcs.assign(grp=fcs.apply(group, axis=1))
        stack = fg.groupby(["grp", "Date"])["p50"].sum().unstack(0).reindex(FC_MONTHS).fillna(0)
        hist12 = h[h["Date"].isin(LAST12)].groupby("SKU_ID")["demand"].sum()
        fc19 = fcs.groupby("SKU_ID")["p50"].sum()
        rule = fcs[fcs["lifecycle_rule"] != ""].groupby("SKU_ID")["lifecycle_rule"].first()
        src = fcs.groupby("SKU_ID")["source"].agg(lambda s: s.value_counts().index[0])
        special = ms[(ms["Lifecycle_Status"] != "Active") | ms["Replaces_SKU"].notna()
                     | (ms["Launch_Date"] >= "2024-01")].copy()
        special["last12"] = special["SKU_ID"].map(hist12).fillna(0)
        special["fc19"] = special["SKU_ID"].map(fc19)
        special["rule"] = special["SKU_ID"].map(rule).astype("string").fillna("")
        special["source"] = special["SKU_ID"].map(src).astype("string").fillna("")
        sku_lines = {}
        for s in special["SKU_ID"]:
            hs = h[h["SKU_ID"] == s].groupby("Date")["demand"].sum().reindex(HIST_MONTHS[-36:])
            fs = fcs[fcs["SKU_ID"] == s].groupby("Date")["p50"].sum().reindex(FC_MONTHS)
            sku_lines[s] = {"hist": hs.to_numpy(), "fc": fs.to_numpy()}
        return clean({"status": [{"status": k, "n": v} for k, v in ms["Lifecycle_Status"].value_counts().items()],
                      "months": FC_MONTHS, "stack": {c: stack[c].to_numpy() for c in stack.columns},
                      "table": records(special.sort_values(["Lifecycle_Status", "SKU_ID"])),
                      "hist_months": HIST_MONTHS[-36:], "lines": sku_lines,
                      "kpi": {"skus": len(ms), "eol_in_horizon": int(((ms["EOL_Date"] >= TEST_START) & (ms["EOL_Date"] <= TEST_END)).sum()),
                              "launch_in_horizon": int(((ms["Launch_Date"] >= TEST_START) & (ms["Launch_Date"] <= TEST_END)).sum()),
                              "discontinued": int((ms["Lifecycle_Status"] == "Discontinued").sum())}})

    # ---------------------------------------------------------------- 10. stockouts & lost sales
    @lru_cache(maxsize=128)  # noqa: B019 (one Store per process)
    def stockouts(self, F: Filters) -> dict:
        h = self.sel(self.hist, F)
        so = h[h["stockout_listed"] | h["stockout_inferred"]]
        m = h.groupby("Date").agg(demand=("demand", "sum"), shipped=("Units_Sold", "sum"),
                                  lost=("lost_units_est", "sum"), back=("Backordered_Units", "sum"),
                                  unserved=("unserved", "sum")).reindex(HIST_MONTHS)
        listed = so[so["stockout_listed"]].groupby("Date").size().reindex(HIST_MONTHS).fillna(0)
        inferred = so[so["stockout_inferred"] & ~so["stockout_listed"]].groupby("Date").size().reindex(HIST_MONTHS).fillna(0)
        reason = so[so["stockout_listed"]].groupby("Reason").agg(events=("SKU_ID", "size"), lost=("lost_units_est", "sum")).reset_index()
        price = h["Unit_Price_USD"]
        so = so.assign(lost_value=so["lost_units_est"].fillna(0) * price.loc[so.index].fillna(0))
        fam = so.groupby("Product_Family").agg(events=("SKU_ID", "size"), lost=("lost_units_est", "sum"),
                                               value=("lost_value", "sum")).reset_index()
        reg = so.groupby("Region").agg(events=("SKU_ID", "size"), lost=("lost_units_est", "sum"),
                                       value=("lost_value", "sum")).reset_index()
        top = so.groupby(["SKU_ID", "Region", "Product_Family"]).agg(
            months=("Date", "nunique"), lost=("lost_units_est", "sum"), value=("lost_value", "sum"),
            last=("Date", "max"), reason=("Reason", lambda s: s.dropna().value_counts().index[0] if s.notna().any() else "inferred")
        ).reset_index().sort_values("lost", ascending=False).head(50)
        fill = (m["shipped"] / m["demand"]).where(m["demand"] > 0)
        return clean({"months": HIST_MONTHS, "lost": m["lost"].to_numpy(), "back": m["back"].to_numpy(),
                      "fill_rate": fill.to_numpy(), "listed": listed.to_numpy(), "inferred": inferred.to_numpy(),
                      "reason": records(reason.sort_values("lost", ascending=False)), "family": records(fam),
                      "region": records(reg), "top": records(top),
                      "kpi": {"events": len(so), "lost_units": float(so["lost_units_est"].sum()),
                              "lost_value": float(so["lost_value"].sum()),
                              "fill_rate": float(m["shipped"].sum() / m["demand"].sum()) if m["demand"].sum() else None,
                              "last12_events": int(so["Date"].isin(LAST12).sum()),
                              "uplift": float(m["demand"].sum() / m["shipped"].sum() - 1) if m["shipped"].sum() else None}})

    # ---------------------------------------------------------------- 11. exception list
    def _series_table(self) -> pd.DataFrame:
        fc = self.fc
        k = SERIES_KEYS
        nxt = fc[fc["Date"].isin(NEXT12)].groupby(k).agg(next12=("p50", "sum"), p10=("p10", "sum"), p90=("p90", "sum"),
                                                          rev12=("Revenue_P50_USD", "sum"))
        info = fc.groupby(k).agg(Product_Family=("Product_Family", "first"), source=("source", "first"),
                                 rule=("lifecycle_rule", lambda s: next((x for x in s if x), "")))
        h = self.hist
        last = h[h["Date"].isin(LAST12)].groupby(k)["demand"].sum().rename("last12")
        recent = HIST_MONTHS[-6:]
        so = h[h["Date"].isin(recent) & (h["stockout_listed"] | h["stockout_inferred"])].groupby(k).size().rename("stockouts6")
        a = self.anom
        sus = a[a["Date"].isin(recent) & a["category"].isin(["suspected"])].groupby(k).size().rename("suspected6")
        shift = a[a["Date"].isin(HIST_MONTHS[-12:]) & a["anomaly_type"].isin(
            ["level_shift_up", "level_shift_down", "customer_switch", "lost_account"])].groupby(k).size().rename("shift12")
        b = self.bt
        acc = b.groupby(k).apply(lambda g: pd.Series({"bt_wape": wape(g["y_demand"], g[FINAL_MODEL]),
                                                      "bt_bias": bias(g["y_demand"], g[FINAL_MODEL]),
                                                      "bt_actual": float(g["y_demand"].sum())}), include_groups=False)
        t = info.join(nxt).join(last).join(so).join(sus).join(shift).join(acc).reset_index()
        t[["last12", "stockouts6", "suspected6", "shift12"]] = t[["last12", "stockouts6", "suspected6", "shift12"]].fillna(0)
        t["change"] = np.where(t["last12"] > 0, t["next12"] / t["last12"] - 1, np.nan)
        t["band_width"] = np.where(t["next12"] > 0, (t["p90"] - t["p10"]) / t["next12"], np.nan)
        flags, actions = [], []
        for r in t.itertuples():
            f, act = [], []
            if r.source == "new_launch_rule":
                f.append("New launch")
                act.append("analog-based forecast; confirm launch date and initial orders with sales")
            if r.rule in ("eol_rundown", "after_eol_zero"):
                f.append("EOL run-down")
                act.append("cap purchase orders; plan last-time buy")
            if r.last12 >= 60 and not math.isnan(r.change) and abs(r.change) >= 0.35 and r.source == "gbm" and not r.rule:
                f.append("Forecast jump" if r.change > 0 else "Forecast drop")
                act.append("check the change against customer plans before committing supply")
            if r.next12 >= 30 and not math.isnan(r.band_width) and r.band_width >= 1.0:
                f.append("Wide range")
                act.append("plan to P90 for critical items; review monthly")
            if r.bt_actual and r.bt_actual >= 100 and r.bt_wape is not None and not math.isnan(r.bt_wape) and r.bt_wape >= 0.6:
                f.append("Low past accuracy")
                act.append("hold extra buffer; consider manual override")
            if r.stockouts6 >= 2:
                f.append("Recent stockouts")
                act.append("history understates demand; secure supply")
            if r.suspected6 >= 2:
                f.append("Unexplained dips")
                act.append("ask the customer whether the dip is real")
            if r.shift12 >= 1:
                f.append("Level shift")
                act.append("forecast anchored on new level; confirm it persists")
            flags.append(f)
            actions.append("; ".join(dict.fromkeys(act)))
        t["flags"] = flags
        t["action"] = actions
        t["n_flags"] = t["flags"].str.len()
        t["priority_score"] = t["n_flags"] * np.log1p(t["rev12"].fillna(0))
        return t[t["n_flags"] > 0].sort_values("priority_score", ascending=False)

    @lru_cache(maxsize=256)  # noqa: B019 (one Store per process)
    def exceptions(self, F: Filters, flag: str = "") -> dict:
        t = self.sel(self._series, F)
        counts = t["flags"].explode().value_counts()
        if flag:
            t = t[t["flags"].apply(lambda f: flag in f)]
        q = t["priority_score"].quantile([0.5, 0.85]) if len(t) else pd.Series([0, 0], index=[0.5, 0.85])
        t = t.assign(priority=np.where(t["priority_score"] >= q[0.85], "High",
                                       np.where(t["priority_score"] >= q[0.5], "Medium", "Low")))
        return clean({"table": records(t.head(400).drop(columns=["p10", "p90"])), "total": len(t),
                      "counts": [{"flag": k, "n": v} for k, v in counts.items()],
                      "rev_at_stake": float(t["rev12"].sum())})

    def exceptions_frame(self, F: Filters) -> pd.DataFrame:
        t = self.sel(self._series, F).copy()
        t["flags"] = t["flags"].str.join(", ")
        return t

    # ---------------------------------------------------------------- 12. downloads
    def download(self, kind: str, F: Filters, window: int = 6) -> pd.DataFrame:
        if kind == "forecast":
            return self.sel(self.fc, F).drop(columns=["lo2", "hi2", "series"])
        if kind == "history":
            return self.sel(self.hist, F)
        if kind == "anomalies":
            return self.sel(self.anom, F)
        if kind == "procurement":
            return self.procurement_table(F, window)
        if kind == "exceptions":
            return self.exceptions_frame(F)
        if kind == "backtest":
            return self.sel(self.bt, F)
        if kind == "submission":
            return pd.read_csv(OUTPUTS / "submission.csv")
        raise KeyError(kind)


_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
