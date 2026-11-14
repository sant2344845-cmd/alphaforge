from __future__ import annotations
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-8

def _pad_front(arr: np.ndarray, window: int) -> np.ndarray:
    pad = np.full((window - 1, arr.shape[1]), np.nan, dtype=np.float64)
    return np.vstack([pad, arr])

def cs_rank(x: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(x)
    counts = np.sum(valid, axis=1, keepdims=True)
    filled = np.where(valid, x, np.inf)
    order = np.argsort(filled, axis=1)
    ranks = np.argsort(order, axis=1).astype(np.float64) + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(counts > 0, ranks / counts, np.nan)
    out[~valid] = np.nan
    return out

def cs_zscore(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, ddof=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > EPS, (x - mean) / std, 0.0)

def cs_demean(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return x - np.nanmean(x, axis=1, keepdims=True)

def ts_delay(x: np.ndarray, d: int) -> np.ndarray:
    if d <= 0:
        return x.copy()
    out = np.full_like(x, np.nan, dtype=np.float64)
    if d < x.shape[0]:
        out[d:] = x[:-d]
    return out

def ts_delta(x: np.ndarray, d: int) -> np.ndarray:
    return x - ts_delay(x, d)

def ts_pct_change(x: np.ndarray, d: int) -> np.ndarray:
    prev = ts_delay(x, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(prev) > EPS, (x - prev) / np.abs(prev), np.nan)

def ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    if d <= 1:
        return x.copy()
    w = sliding_window_view(x, window_shape=d, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _pad_front(np.nanmean(w, axis=-1), d)

def ts_std(x: np.ndarray, d: int) -> np.ndarray:
    if d <= 1:
        return np.zeros_like(x)
    w = sliding_window_view(x, window_shape=d, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _pad_front(np.nanstd(w, axis=-1, ddof=1), d)

def ts_zscore(x: np.ndarray, d: int) -> np.ndarray:
    std = ts_std(x, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > EPS, (x - ts_mean(x, d)) / std, 0.0)

def ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    if d <= 1:
        return np.full_like(x, 0.5)
    w = sliding_window_view(x, window_shape=d, axis=0)
    cur = w[:, :, -1:]
    valid = ~np.isnan(w)
    cnt = np.sum(valid, axis=-1)
    less = np.sum(valid & (w < cur), axis=-1)
    eq = np.sum(valid & (w == cur), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((cnt > 1) & ~np.isnan(w[:, :, -1]), (less + 0.5 * (eq - 1.0)) / (cnt - 1.0), np.nan)
    return _pad_front(r, d)

def ts_decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    if d <= 1:
        return x.copy()
    w = sliding_window_view(x, window_shape=d, axis=0)
    weights = np.arange(1, d + 1, dtype=np.float64)[None, None, :]
    valid = ~np.isnan(w)
    w_sum = np.sum(np.where(valid, weights, 0.0), axis=-1)
    val_sum = np.sum(np.where(valid, w * weights, 0.0), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(w_sum > EPS, val_sum / w_sum, np.nan)
    return _pad_front(res, d)

def ts_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    if d <= 2:
        return np.full_like(x, np.nan)
    wx = sliding_window_view(x, window_shape=d, axis=0)
    wy = sliding_window_view(y, window_shape=d, axis=0)
    valid = (~np.isnan(wx)) & (~np.isnan(wy))
    n_valid = np.sum(valid, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dx = np.where(valid, wx - np.nanmean(np.where(valid, wx, np.nan), axis=-1, keepdims=True), 0.0)
        dy = np.where(valid, wy - np.nanmean(np.where(valid, wy, np.nan), axis=-1, keepdims=True), 0.0)
    cov = np.sum(dx * dy, axis=-1)
    denom = np.sqrt(np.sum(dx ** 2, axis=-1) * np.sum(dy ** 2, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where((n_valid >= 3) & (denom > EPS), cov / denom, np.nan)
    return _pad_front(corr, d)

def safe_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(y) > EPS, x / y, np.nan)

OPERATOR_REGISTRY: dict[str, dict] = {
    "cs_rank": {"fn": cs_rank, "arity": 1, "params": []},
    "cs_zscore": {"fn": cs_zscore, "arity": 1, "params": []},
    "cs_demean": {"fn": cs_demean, "arity": 1, "params": []},
    "neg": {"fn": lambda x: -x, "arity": 1, "params": []},
    "abs": {"fn": np.abs, "arity": 1, "params": []},
    "sign": {"fn": np.sign, "arity": 1, "params": []},
    "ts_delay": {"fn": ts_delay, "arity": 1, "params": ["d"]},
    "ts_delta": {"fn": ts_delta, "arity": 1, "params": ["d"]},
    "ts_pct_change": {"fn": ts_pct_change, "arity": 1, "params": ["d"]},
    "ts_mean": {"fn": ts_mean, "arity": 1, "params": ["d"]},
    "ts_std": {"fn": ts_std, "arity": 1, "params": ["d"]},
    "ts_zscore": {"fn": ts_zscore, "arity": 1, "params": ["d"]},
    "ts_rank": {"fn": ts_rank, "arity": 1, "params": ["d"]},
    "ts_decay_linear": {"fn": ts_decay_linear, "arity": 1, "params": ["d"]},
    "add": {"fn": np.add, "arity": 2, "params": []},
    "sub": {"fn": np.subtract, "arity": 2, "params": []},
    "mul": {"fn": np.multiply, "arity": 2, "params": []},
    "div": {"fn": safe_div, "arity": 2, "params": []},
    "ts_corr": {"fn": ts_corr, "arity": 2, "params": ["d"]},
}
