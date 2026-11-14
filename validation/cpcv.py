from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy.stats import kurtosis, norm, skew
from engine.operators import EPS
from validation.metrics import evaluate_alpha_performance

EULER_MASCHERONI = 0.5772156649015329

def deflated_sharpe_ratio(
    returns: np.ndarray,
    num_trials: int,
    annualization: int = 252,
) -> float:
    """
    Calcola il Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    Restituisce la probabilità [0, 1] che lo Sharpe Ratio sia genuino dopo
    aver corretto per multiple testing (num_trials), skewness e kurtosis.
    """
    clean = returns[~np.isnan(returns)]
    t_len = len(clean)
    if t_len < 30 or num_trials < 1:
        return 0.0

    std_r = float(np.std(clean, ddof=1))
    if std_r <= EPS:
        return 0.0

    # Sharpe non annualizzato (frequenza nativa)
    sr_hat = float(np.mean(clean)) / std_r

    if num_trials == 1:
        sr_benchmark = 0.0
    else:
        # Valore atteso del massimo di num_trials variabili normali standard
        z1 = norm.ppf(1.0 - 1.0 / num_trials)
        z2 = norm.ppf(1.0 - 1.0 / (num_trials * np.e))
        e_max_z = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
        # Assumiamo deviazione standard cross-trial conservativa pari a 1/sqrt(T)
        sr_benchmark = e_max_z / np.sqrt(t_len)

    g3 = float(skew(clean, bias=False))
    g4 = float(kurtosis(clean, fisher=False, bias=False))  # Pearson kurtosis (normale = 3.0)

    denom_sq = 1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * (sr_hat ** 2)
    if denom_sq <= EPS:
        return 0.0

    z_stat = ((sr_hat - sr_benchmark) * np.sqrt(t_len - 1.0)) / np.sqrt(denom_sq)
    return float(norm.cdf(z_stat))

def run_cpcv_validation(
    signal: np.ndarray,
    fwd_returns: np.ndarray,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_days: int = 5,
    cost_bps: float = 5.0,
) -> dict[str, float]:
    """
    Combinatorial Purged Cross-Validation (CPCV) con Purge & Embargo.
    Valuta il segnale su tutte le C(n_splits, n_test_splits) partizioni combinatorie.
    """
    t_len = signal.shape[0]
    indices = np.arange(t_len)
    folds = np.array_split(indices, n_splits)

    oos_sharpes: list[float] = []
    oos_ics: list[float] = []

    for test_fold_ids in combinations(range(n_splits), n_test_splits):
        test_mask = np.zeros(t_len, dtype=bool)
        for fid in test_fold_ids:
            start_idx = max(0, folds[fid][0] + embargo_days)
            end_idx = max(start_idx, folds[fid][-1] - embargo_days + 1)
            test_mask[start_idx:end_idx] = True

        if np.sum(test_mask) < 30:
            continue

        metrics = evaluate_alpha_performance(
            signal[test_mask], fwd_returns[test_mask], cost_bps=cost_bps
        )
        oos_sharpes.append(metrics.sharpe_net)
        oos_ics.append(metrics.mean_ic)

    if not oos_sharpes:
        return {"cpcv_mean_sharpe": 0.0, "cpcv_min_sharpe": 0.0, "cpcv_pct_positive": 0.0, "cpcv_mean_ic": 0.0}

    arr_sr = np.array(oos_sharpes)
    return {
        "cpcv_mean_sharpe": float(np.mean(arr_sr)),
        "cpcv_min_sharpe": float(np.min(arr_sr)),
        "cpcv_pct_positive": float(np.mean(arr_sr > 0.0)),
        "cpcv_mean_ic": float(np.mean(oos_ics)),
    }
