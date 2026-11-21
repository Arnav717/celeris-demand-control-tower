"""Global LightGBM, direct multi-horizon.

Target is scaled by the series' 12-month mean at the origin (y / scale) and rows are
weighted by scale, so an absolute-error objective on the scaled target equals absolute
error in units: the loss lines up with WAPE while one model serves all series sizes.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from celeris.features.build import CATS, feature_columns

BASE_PARAMS = dict(
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1,
    num_threads=0, seed=7,
)


def _dataset(df: pd.DataFrame, cols: list[str]):
    scale = df["_scale"].to_numpy()
    y = df["y"].to_numpy() / scale
    w = df["w"].to_numpy() * scale
    return lgb.Dataset(df[cols], y, weight=w, categorical_feature=[c for c in CATS if c in cols],
                       free_raw_data=True)


class GBM:
    def __init__(self, objective: str = "l1", alpha: float | None = None, rounds: int = 700,
                 params: dict | None = None):
        self.params = {**BASE_PARAMS, **(params or {}), "objective": objective}
        if alpha is not None:
            self.params["alpha"] = alpha
        self.rounds = rounds
        self.model: lgb.Booster | None = None
        self.cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> GBM:
        self.cols = feature_columns(df)
        self.model = lgb.train(self.params, _dataset(df, self.cols), num_boost_round=self.rounds)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(df[self.cols], num_threads=0) * df["_scale"].to_numpy()
        return np.clip(pred, 0, None)

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.cols,
                             "gain": self.model.feature_importance("gain")}).sort_values("gain", ascending=False)
