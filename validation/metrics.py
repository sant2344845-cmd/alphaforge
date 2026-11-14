from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from engine.operators import EPS, cs_demean

@dataclass
class BacktestMetrics:
    mean_ic: float
    ic_ir: float
    ic_tstat: float
    sharpe_gross: float
    sharpe_net: float
    annual_return_net: float
    annual_vol: float
    max_drawdown: float
    daily_turnover: float
    net_returns: np.ndarray
    ic_series: np.ndarray

def _fast_row_rank(mat: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(mat)
    filled = np.where(valid, mat, np.inf)
    order = np.argsort(filled, axis=1)
    ranks = np.argsort(order, axis=1).astype(np.float64) + 1.0
    ranks[~valid] = np.nan
    return ranks

def signal_to_weights(signal: np.ndarray) -> np.ndarray:
    ranks = _fast_row_rank(signal)
    centered = cs_demean(ranks)
    gross = np.nansum(np.abs(centered), axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(gross > EPS, centered / gross, 0.0)
    return np.nan_to_num(weights, nan=0.0)

def compute_rank_ic(signal: np.ndarray, fwd_returns: np.ndarray) -> np.ndarray:
    valid = (~np.isnan(signal)) & (~np.isnan(fwd_returns))
    counts = np.sum(valid, axis=1)
    s_clean = np.where(valid, signal, np.nan)
    r_clean = np.where(valid, fwd_returns, np.nan)
    rs = _fast_row_rank(s_clean)
    rr = _fast_row_rank(r_clean)
    rs_dm = rs - np.nanmean(rs, axis=1, keepdims=True)
    rr_dm = rr - np.nanmean(rr, axis=1, keepdims=True)
    num = np.nansum(rs_dm * rr_dm, axis=1)
    den = np.sqrt(np.nansum(rs_dm ** 2, axis=1) * np.nansum(rr_dm ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.where((counts >= 5) & (den > EPS), num / den, np.nan)
    return ic

def evaluate_alpha_performance(signal: np.ndarray, fwd_returns: np.ndarray, cost_bps: float = 3.0, annualization: int = 252) -> BacktestMetrics:
    weights = signal_to_weights(signal)
    clean_ret = np.nan_to_num(fwd_returns, nan=0.0)
    gross_ret = np.sum(weights * clean_ret, axis=1)
    turnover = 0.5 * np.sum(np.abs(np.diff(weights, axis=0, prepend=0.0)), axis=1)
    net_ret = gross_ret - turnover * (cost_bps * 1e-4)

    active = np.sum(np.abs(weights), axis=1) > EPS
    if np.sum(active) < 20:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, net_ret, np.array([]))

    g_act, n_act, t_act = gross_ret[active], net_ret[active], turnover[active]
    ic_series = compute_rank_ic(signal, fwd_returns)
    v_ic = ic_series[~np.isnan(ic_series)]
    mean_ic = float(np.mean(v_ic)) if len(v_ic) > 0 else 0.0
    std_ic = float(np.std(v_ic, ddof=1)) if len(v_ic) > 1 else 0.0
    ic_ir = mean_ic / std_ic if std_ic > EPS else 0.0
    ic_tstat = ic_ir * np.sqrt(len(v_ic)) if len(v_ic) > 1 else 0.0

    std_g, std_n = float(np.std(g_act, ddof=1)), float(np.std(n_act, ddof=1))
    sr_g = (float(np.mean(g_act)) / std_g) * np.sqrt(annualization) if std_g > EPS else 0.0
    sr_n = (float(np.mean(n_act)) / std_n) * np.sqrt(annualization) if std_n > EPS else 0.0
    cum = np.cumprod(1.0 + n_act)
    max_dd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)))

    return BacktestMetrics(mean_ic, ic_ir, ic_tstat, sr_g, sr_n, float(np.mean(n_act)) * annualization, std_n * np.sqrt(annualization), max_dd, float(np.mean(t_act)), net_ret, ic_series)
