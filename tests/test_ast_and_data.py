import numpy as np, pytest
from data.loader import build_dataset_from_matrices
from engine.ast_parser import AlphaSyntaxError, parse_alpha
from engine.operators import cs_rank, ts_corr

def _ds():
    rng = np.random.default_rng(123)
    b = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=(50, 10)), axis=0)
    return build_dataset_from_matrices(np.array([f"2025-01-{i+1:02d}" for i in range(50)]), [f"S{i}" for i in range(10)], b, b+1, b-1, b+0.2, rng.uniform(1e6, 5e6, (50, 10)))

def test_ast_eval():
    ds = _ds()
    node = parse_alpha("-cs_rank(ts_corr(close, volume, 10))")
    np.testing.assert_allclose(node.evaluate(ds.to_feature_dict()), -cs_rank(ts_corr(ds.close, ds.volume, 10)), equal_nan=True)

def test_ast_security():
    with pytest.raises(AlphaSyntaxError): parse_alpha("__import__('os').system('ls')")
    with pytest.raises(AlphaSyntaxError): parse_alpha("cs_rank(fwd_ret_1d)")
    with pytest.raises(AlphaSyntaxError): parse_alpha("ts_mean(close, -5)")

def test_fwd_alignment():
    ds = _ds()
    assert ds.fwd_ret_1d[0, 0] == pytest.approx((ds.open[2, 0] - ds.open[1, 0]) / ds.open[1, 0])
    assert np.all(np.isnan(ds.fwd_ret_1d[-2:]))
