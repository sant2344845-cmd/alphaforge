import numpy as np
import pytest
from engine.operators import OPERATOR_REGISTRY, cs_demean, cs_rank, cs_zscore, ts_corr, ts_decay_linear

@pytest.fixture
def sample_matrices():
    rng = np.random.default_rng(42)
    x = rng.normal(100.0, 5.0, size=(60, 15))
    y = rng.normal(1_000_000.0, 200_000.0, size=(60, 15))
    x[0:5, 0] = np.nan
    x[25, 7] = np.nan
    y[10:12, 3] = np.nan
    return x, y

def test_zero_lookahead_bias_all_operators(sample_matrices):
    x, y = sample_matrices
    t_cut, window = 40, 10
    x_shock, y_shock = x.copy(), y.copy()
    x_shock[t_cut:] *= 1000.0
    y_shock[t_cut:] = -99999.0

    for name, spec in OPERATOR_REGISTRY.items():
        fn, arity, has_d = spec["fn"], spec["arity"], "d" in spec["params"]
        if arity == 1 and not has_d:
            r1, r2 = fn(x)[:t_cut], fn(x_shock)[:t_cut]
        elif arity == 1 and has_d:
            r1, r2 = fn(x, window)[:t_cut], fn(x_shock, window)[:t_cut]
        elif arity == 2 and not has_d:
            r1, r2 = fn(x, y)[:t_cut], fn(x_shock, y_shock)[:t_cut]
        else:
            r1, r2 = fn(x, y, window)[:t_cut], fn(x_shock, y_shock, window)[:t_cut]
        np.testing.assert_allclose(r1, r2, rtol=1e-12, atol=1e-12, equal_nan=True)

def test_cs_operators():
    x = np.array([[10.0, 20.0, 30.0, np.nan]])
    r = cs_rank(x)
    assert np.isnan(r[0, 3])
    np.testing.assert_allclose(r[0, :3], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(np.nanmean(cs_demean(x), axis=1), [0.0], atol=1e-12)
    np.testing.assert_allclose(np.nanstd(cs_zscore(x), axis=1), [1.0], atol=1e-12)

def test_ts_decay_and_corr():
    x = np.array([[1.0], [2.0], [3.0]])
    assert ts_decay_linear(x, 3)[2, 0] == pytest.approx(14.0 / 6.0)
    t = np.arange(20, dtype=np.float64).reshape(-1, 1)
    np.testing.assert_allclose(ts_corr(t, 2.0 * t + 5.0, d=5)[4:], 1.0, atol=1e-12)
