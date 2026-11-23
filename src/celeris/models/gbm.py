"""Global LightGBM, direct multi-horizon.

Target is scaled by the series' 12-month mean at the origin (y / scale) and rows are
weighted by scale, so an absolute-error objective on the scaled target equals absolute
error in units: the loss lines up with WAPE while one model serves all series sizes.

`seeds` > 1 averages models trained with different row / feature subsampling seeds
(variance reduction: -0.001 WAPE in the backtest, better or equal at every origin).
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from celeris.features.build import CATS, feature_columns

# capped by default: all-core turbo on laptops can trip power / thermal limits
THREADS = int(os.environ.get("CELERIS_THREADS", "4"))

BASE_PARAMS = {
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "verbose": -1,
    "num_threads": THREADS, "seed": 7,
}


def _dataset(df: pd.DataFrame, cols: list[str]):
    scale = df["_scale"].to_numpy()
    y = df["y"].to_numpy() / scale
    w = df["w"].to_numpy() * scale
    return lgb.Dataset(df[cols], y, weight=w, categorical_feature=[c for c in CATS if c in cols],
                       free_raw_data=True)


class GBM:
    def __init__(self, objective: str = "l1", alpha: float | None = None, rounds: int = 700,
                 params: dict | None = None, seeds: tuple[int, ...] = (7,)):
        self.params = {**BASE_PARAMS, **(params or {}), "objective": objective}
        if alpha is not None:
            self.params["alpha"] = alpha
        self.rounds = rounds
        self.seeds = seeds
        self.models: list[lgb.Booster] = []
        self.cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> GBM:
        self.cols = feature_columns(df)
        self.models = [lgb.train({**self.params, "seed": s}, _dataset(df, self.cols), num_boost_round=self.rounds)
                       for s in self.seeds]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = np.mean([m.predict(df[self.cols], num_threads=THREADS) for m in self.models], axis=0)
        return np.clip(raw * df["_scale"].to_numpy(), 0, None)

    def importance(self) -> pd.DataFrame:
        gain = np.mean([m.feature_importance("gain") for m in self.models], axis=0)
        return pd.DataFrame({"feature": self.cols, "gain": gain}).sort_values("gain", ascending=False)
