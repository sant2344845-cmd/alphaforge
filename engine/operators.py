"""
engine/operators.py
Operatori vettorizzati per Alpha Research su matrici 2D (T, N).
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

EPS = 1e-8

def safe_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(y) > EPS, x / y, np.nan)

def _rolling_window(arr: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError(f"Window must be >= 1, got {window}")
    return sliding_window_view(arr, window_shape=window, axis=0)

def cs_rank(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        valid = ~np.isnan(row)
        count = np.sum(valid)
        if count > 1:
            ranks = rankdata(row[valid], method="average")
            out[t, valid] = (ranks - 1.0) / (count - 1.0)
        elif count == 1:
            out[t, valid] = 0.5
    return out

def cs_zscore(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    return safe_div(x - mean, std)

def cs_demean(x: np.ndarray) -> np.ndarray:
    return x - np.nanmean(x, axis=1, keepdims=True)

def ts_delay(x: np.ndarray, d: int) -> np.ndarray:
    if d == 0:
        return x.copy()
    if d < 0:
        raise ValueError("Delay d must be >= 0")
    out = np.full_like(x, np.nan, dtype=np.float64)
    if d < x.shape[0]:
        out[d:] = x[:-d]
    return out

def ts_delta(x: np.ndarray, d: int) -> np.ndarray:
    return x - ts_delay(x, d)

def ts_pct_change(x: np.ndarray, d: int) -> np.ndarray:
    prev = ts_delay(x, d)
    return safe_div(x - prev, np.abs(prev))

def ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    if d == 1:
        return x.copy()
    out = np.full_like(x, np.nan, dtype=np.float64)
    if x.shape[0] >= d:
        out[d - 1 :] = np.nanmean(_rolling_window(x, d), axis=-1)
    return out

def ts_std(x: np.ndarray, d: int) -> np.ndarray:
    if d < 2:
        raise ValueError("ts_std requires d >= 2")
    out = np.full_like(x, np.nan, dtype=np.float64)
    if x.shape[0] >= d:
        out[d - 1 :] = np.nanstd(_rolling_window(x, d), axis=-1, ddof=1)
    return out

def ts_zscore(x: np.ndarray, d: int) -> np.ndarray:
    return safe_div(x - ts_mean(x, d), ts_std(x, d))

def ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    if d < 2:
        raise ValueError("ts_rank requires d >= 2")
    out = np.full_like(x, np.nan, dtype=np.float64)
    if x.shape[0] >= d:
        win = _rolling_window(x, d)
        current = win[:, :, -1:]
        valid_mask = ~np.isnan(win)
        valid_counts = np.sum(valid_mask, axis=-1)
        less_equal = np.sum((win <= current) & valid_mask, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ranks = np.where(
                (valid_counts > 1) & (~np.isnan(win[:, :, -1])),
                (less_equal - 1.0) / (valid_counts - 1.0),
                np.nan,
            )
        out[d - 1 :] = ranks
    return out

def ts_decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    if d == 1:
        return x.copy()
    out = np.full_like(x, np.nan, dtype=np.float64)
    if x.shape[0] >= d:
        weights = np.arange(1, d + 1, dtype=np.float64)
        weights /= weights.sum()
        win = _rolling_window(x, d)
        valid = ~np.isnan(win)
        weighted_vals = np.where(valid, win * weights, 0.0)
        weight_sums = np.sum(np.where(valid, weights, 0.0), axis=-1)
        out[d - 1 :] = safe_div(np.sum(weighted_vals, axis=-1), weight_sums)
    return out

def ts_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    if d < 3:
        raise ValueError("ts_corr requires d >= 3")
    out = np.full_like(x, np.nan, dtype=np.float64)
    if x.shape[0] >= d:
        wx = _rolling_window(x, d)
        wy = _rolling_window(y, d)
        valid = (~np.isnan(wx)) & (~np.isnan(wy))
        n_valid = np.sum(valid, axis=-1)
        wx_v = np.where(valid, wx, np.nan)
        wy_v = np.where(valid, wy, np.nan)
        dx = np.where(valid, wx_v - np.nanmean(wx_v, axis=-1, keepdims=True), 0.0)
        dy = np.where(valid, wy_v - np.nanmean(wy_v, axis=-1, keepdims=True), 0.0)
        cov = np.sum(dx * dy, axis=-1)
        denom = np.sqrt(np.sum(dx * dx, axis=-1) * np.sum(dy * dy, axis=-1))
        corr = np.where((n_valid >= 3) & (denom > EPS), cov / denom, np.nan)
        out[d - 1 :] = np.clip(corr, -1.0, 1.0)
    return out

OPERATOR_REGISTRY: dict[str, dict] = {
    "cs_rank": {"fn": cs_rank, "arity": 1, "params": []},
    "cs_zscore": {"fn": cs_zscore, "arity": 1, "params": []},
    "cs_demean": {"fn": cs_demean, "arity": 1, "params": []},
    "ts_delay": {"fn": ts_delay, "arity": 1, "params": ["d"]},
    "ts_delta": {"fn": ts_delta, "arity": 1, "params": ["d"]},
    "ts_pct_change": {"fn": ts_pct_change, "arity": 1, "params": ["d"]},
    "ts_mean": {"fn": ts_mean, "arity": 1, "params": ["d"]},
    "ts_std": {"fn": ts_std, "arity": 1, "params": ["d"]},
    "ts_zscore": {"fn": ts_zscore, "arity": 1, "params": ["d"]},
    "ts_rank": {"fn": ts_rank, "arity": 1, "params": ["d"]},
    "ts_decay_linear": {"fn": ts_decay_linear, "arity": 1, "params": ["d"]},
    "ts_corr": {"fn": ts_corr, "arity": 2, "params": ["d"]},
    "add": {"fn": np.add, "arity": 2, "params": []},
    "sub": {"fn": np.subtract, "arity": 2, "params": []},
    "mul": {"fn": np.multiply, "arity": 2, "params": []},
    "div": {"fn": safe_div, "arity": 2, "params": []},
    "neg": {"fn": np.negative, "arity": 1, "params": []},
    "abs": {"fn": np.abs, "arity": 1, "params": []},
    "sign": {"fn": np.sign, "arity": 1, "params": []},
    "log": {"fn": lambda x: np.log(np.maximum(x, EPS)), "arity": 1, "params": []},
}
