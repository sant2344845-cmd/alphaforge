from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from agents.researcher import generate_hypotheses
from data.loader import load_us_equities
from engine.genetic import EvolutionaryAlphaEngine

def main() -> None:
    print("[1/3] Caricamento universo US Equities (2018-2026)...")
    ds = load_us_equities(start="2018-01-01", end="2026-01-01")
    print(f"      Dataset pronto: {len(ds.dates)} giorni x {len(ds.tickers)} titoli.")

    print("[2/3] Generazione ipotesi iniziali (Agente LLM / Quant Seeds)...")
    seeds = generate_hypotheses(n_hypotheses=8)
    print(f"      {len(seeds)} ipotesi valide pronte per l'evoluzione.")

    print("[3/3] Avvio Evoluzione Genetica & Validazione CPCV/DSR...")
    engine = EvolutionaryAlphaEngine(ds.to_feature_dict(), ds.fwd_ret_1d, cost_bps=5.0, max_corr_threshold=0.40, seed=42)
    zoo = engine.evolve(seed_hypotheses=seeds, generations=5, pop_size=30, top_k_zoo=5)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    summary = []
    returns_dict = {"dates": ds.dates}

    print("\n=== ALPHA ZOO SCOPERTO (ORTOGONALE & VALIDATO) ===")
    for idx, c in enumerate(zoo, start=1):
        print(f"\n#{idx} | {c.formula}")
        print(f"    Razionale : {c.rationale}")
        print(f"    Sharpe Net: {c.metrics.sharpe_net:.2f} | Rank IC: {c.metrics.mean_ic:.4f} (t={c.metrics.ic_tstat:.2f}) | Turnover: {c.metrics.daily_turnover:.1%}")
        print(f"    DSR Prob  : {c.dsr:.1%} | CPCV Mean SR: {c.cpcv_mean_sharpe:.2f} (Pos Folds: {c.cpcv_pct_positive:.0%})")
        summary.append({
            "rank": idx, "formula": c.formula, "rationale": c.rationale,
            "sharpe_net": round(c.metrics.sharpe_net, 3), "sharpe_gross": round(c.metrics.sharpe_gross, 3),
            "mean_ic": round(c.metrics.mean_ic, 5), "ic_ir": round(c.metrics.ic_ir, 3),
            "ic_tstat": round(c.metrics.ic_tstat, 2), "daily_turnover": round(c.metrics.daily_turnover, 4),
            "max_drawdown": round(c.metrics.max_drawdown, 4), "dsr": round(c.dsr, 4),
            "cpcv_mean_sharpe": round(c.cpcv_mean_sharpe, 3), "cpcv_pct_positive": round(c.cpcv_pct_positive, 3),
        })
        returns_dict[f"alpha_{idx}"] = c.metrics.net_returns

    with open(out_dir / "alpha_zoo.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(out_dir / "alpha_returns.npz", **returns_dict)
    print(f"\nRisultati salvati in {out_dir}/alpha_zoo.json")

if __name__ == "__main__":
    main()
