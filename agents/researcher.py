from __future__ import annotations
import os
from pydantic import BaseModel, Field
from engine.ast_parser import ALLOWED_FEATURES, AlphaSyntaxError, parse_alpha
from engine.operators import OPERATOR_REGISTRY

class AlphaHypothesis(BaseModel):
    name: str = Field(description="Nome breve")
    economic_rationale: str = Field(description="Razionale economico")
    formula: str = Field(description="Formula valida AST")

class HypothesisBatch(BaseModel):
    hypotheses: list[AlphaHypothesis]

BUILTIN_QUANT_SEEDS: list[tuple[str, str]] = [
    ("-cs_rank(ts_corr(close, volume, 10))", "Volume-Price Divergence"),
    ("-cs_rank(ts_delta(close, 5))", "Short-Term Reversal 5d"),
    ("cs_rank(div(sub(close, open), add(sub(high, low), 0.001)))", "Intraday Buying Pressure"),
    ("-cs_rank(mul(cs_rank(ts_pct_change(close, 3)), cs_rank(div(volume, adv20))))", "High-Volume Exhaustion"),
    ("cs_rank(ts_decay_linear(sub(vwap, close), 10))", "VWAP Mean Reversion"),
    ("-cs_rank(ts_std(returns, 20))", "Low-Volatility Anomaly"),
    ("-cs_rank(ts_zscore(close, 20))", "Bollinger Mean Reversion"),
]

SYSTEM_PROMPT = f"Senior Quant Researcher. Genera Alpha cross-sectional US equities usando SOLO feature {sorted(ALLOWED_FEATURES)} e operatori {sorted(OPERATOR_REGISTRY.keys())}."

def generate_hypotheses(n_hypotheses: int = 6, model_name: str = "gemini-2.5-flash") -> list[tuple[str, str]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    valid: list[tuple[str, str]] = []
    if api_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model_name,
                contents=f"Genera {n_hypotheses} formule Alpha originali e ortogonali.",
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, response_mime_type="application/json", response_schema=HypothesisBatch, temperature=0.7),
            )
            if resp.parsed and isinstance(resp.parsed, HypothesisBatch):
                for h in resp.parsed.hypotheses:
                    try:
                        valid.append((parse_alpha(h.formula).to_formula(), f"{h.name}: {h.economic_rationale}"))
                    except AlphaSyntaxError:
                        continue
        except Exception:
            pass
    for f, r in BUILTIN_QUANT_SEEDS:
        if len(valid) >= max(n_hypotheses, len(BUILTIN_QUANT_SEEDS)):
            break
        valid.append((f, r))
    return valid
