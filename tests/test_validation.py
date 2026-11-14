import numpy as np, pytest
from validation.cpcv import deflated_sharpe_ratio, run_cpcv_validation
from validation.metrics import evaluate_alpha_performance

def test_positive_alpha_and_turnover_cost():
    rng = np.random.default_rng(7)
    t_len, n_len = 400, 30
    fwd_ret = rng.normal(0.0, 0.015, size=(t_len, n_len))
    # Segnale con vero potere predittivo (IC ~ 0.15)
    signal = fwd_ret + rng.normal(0.0, 0.08, size=(t_len, n_len))

    m_low_cost = evaluate_alpha_performance(signal, fwd_ret, cost_bps=1.0)
    m_high_cost = evaluate_alpha_performance(signal, fwd_ret, cost_bps=25.0)

    assert m_low_cost.mean_ic > 0.10
    assert m_low_cost.sharpe_net > 2.0
    assert m_low_cost.sharpe_net > m_high_cost.sharpe_net
    assert 0.0 < m_low_cost.daily_turnover <= 1.0

def test_deflated_sharpe_penalizes_multiple_testing():
    rng = np.random.default_rng(99)
    # Rumore puro leggermente fortunato
    noisy_ret = rng.normal(0.0005, 0.01, size=500)
    dsr_1 = deflated_sharpe_ratio(noisy_ret, num_trials=1)
    dsr_1000 = deflated_sharpe_ratio(noisy_ret, num_trials=1000)
    assert dsr_1 > dsr_1000
    assert dsr_1000 < 0.50

def test_cpcv_combinations():
    rng = np.random.default_rng(11)
    fwd_ret = rng.normal(0.0, 0.015, size=(600, 25))
    signal = fwd_ret + rng.normal(0.0, 0.09, size=(600, 25))
    res = run_cpcv_validation(signal, fwd_ret, n_splits=6, n_test_splits=2)
    # C(6, 2) = 15 combinazioni
    assert res["cpcv_mean_sharpe"] > 1.0
    assert res["cpcv_pct_positive"] >= 0.80
