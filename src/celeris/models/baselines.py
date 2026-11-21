"""Statistical baselines, vectorised over all series. Each returns (S, H) forecasts
for months t+1 .. t+H using data up to origin t only."""
from __future__ import annotations

import warnings

import numpy as np


def _hist(Y: np.ndarray, t: int) -> np.ndarray:
    return Y[:, : t + 1]


def seasonal_naive(Y: np.ndarray, t: int, H: int = 19) -> np.ndarray:
    """Same month of the most recent available year; fallback = 12m mean."""
    S = Y.shape[0]
    out = np.full((S, H), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fb = np.nanmean(Y[:, max(0, t - 11): t + 1], axis=1)
    for h in range(1, H + 1):
        m = t + h
        k = 12 * int(np.ceil(h / 12))
        out[:, h - 1] = Y[:, m - k] if m - k >= 0 else np.nan
    return np.where(np.isnan(out), fb[:, None], out)


def moving_average(Y: np.ndarray, t: int, H: int = 19, w: int = 12) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(Y[:, max(0, t - w + 1): t + 1], axis=1)
    return np.repeat(mu[:, None], H, axis=1)


def seasonal_index_level(Y: np.ndarray, t: int, H: int = 19, years: int = 3) -> np.ndarray:
    """Level (12m mean) x series seasonal index (same-month mean / overall mean, last N years)."""
    S = Y.shape[0]
    lo = max(0, t - 12 * years + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmean(Y[:, max(0, t - 11): t + 1], axis=1)
        overall = np.nanmean(Y[:, lo: t + 1], axis=1)
        out = np.empty((S, H))
        for h in range(1, H + 1):
            m = t + h
            idx = [j for j in range(m % 12, t + 1, 12) if j >= lo]
            sm = np.nanmean(Y[:, idx], axis=1) if idx else np.full(S, np.nan)
            si = np.where(overall > 0, sm / overall, 1.0)
            out[:, h - 1] = level * np.nan_to_num(si, nan=1.0)
    return np.nan_to_num(out, nan=0.0)


def tsb(Y: np.ndarray, t: int, H: int = 19, a_p: float = 0.1, a_z: float = 0.1) -> np.ndarray:
    """Teunter-Syntetos-Babai: smoothed demand probability x smoothed non-zero size."""
    S = Y.shape[0]
    p = np.full(S, np.nan)
    z = np.full(S, np.nan)
    for j in range(t + 1):
        y = Y[:, j]
        obs = ~np.isnan(y)
        occ = (y > 0) & obs
        init = obs & np.isnan(p)
        p[init] = occ[init].astype(float)
        z[init & occ] = y[init & occ]
        upd = obs & ~init
        p[upd] = p[upd] + a_p * (occ[upd] - p[upd])
        zu = upd & occ
        z[zu] = np.where(np.isnan(z[zu]), y[zu], z[zu] + a_z * (y[zu] - z[zu]))
    f = np.nan_to_num(p * z, nan=0.0)
    return np.repeat(f[:, None], H, axis=1)


BASELINES = {
    "seasonal_naive": seasonal_naive,
    "moving_average_12": moving_average,
    "seasonal_index_x_level": seasonal_index_level,
    "tsb": tsb,
}
