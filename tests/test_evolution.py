import numpy as np
from agents.researcher import generate_hypotheses
from data.loader import build_dataset_from_matrices
from engine.genetic import EvolutionaryAlphaEngine

def test_evolution_e2e():
    rng = np.random.default_rng(2026)
    t_len, n_len = 350, 25
    noise = rng.normal(0.0, 0.015, size=(t_len, n_len))
    prices = np.zeros((t_len, n_len))
    prices[0] = 100.0
    for t in range(1, t_len):
        rev = -0.15 * (prices[t - 1] - prices[max(0, t - 4)]) / prices[max(0, t - 4)] if t > 3 else 0.0
        prices[t] = prices[t - 1] * (1.0 + rev + noise[t])
    ds = build_dataset_from_matrices(np.array(["2025-01-01"] * t_len), [f"S{i}" for i in range(n_len)], prices, prices * 1.01, prices * 0.99, prices, rng.uniform(1e6, 5e6, (t_len, n_len)))
    seeds = generate_hypotheses(6)
    zoo = EvolutionaryAlphaEngine(ds.to_feature_dict(), ds.fwd_ret_1d, cost_bps=2.0, seed=42).evolve(seeds, generations=3, pop_size=16, top_k_zoo=3)
    assert len(zoo) >= 1 and zoo[0].metrics.sharpe_net > 0.5
