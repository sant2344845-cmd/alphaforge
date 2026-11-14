from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random
import numpy as np
from engine.ast_parser import ALLOWED_FEATURES, AlphaNode, AlphaSyntaxError, parse_alpha
from validation.cpcv import deflated_sharpe_ratio, run_cpcv_validation
from validation.metrics import BacktestMetrics, evaluate_alpha_performance

SAFE_WINDOWS = [5, 10, 15, 20, 30, 60]
UNARY_OPS = ["cs_rank", "cs_zscore", "cs_demean", "neg", "abs", "sign"]
TS_UNARY_OPS = ["ts_delta", "ts_pct_change", "ts_mean", "ts_std", "ts_zscore", "ts_rank", "ts_decay_linear"]
BINARY_OPS = ["add", "sub", "mul", "div"]

@dataclass
class AlphaCandidate:
    formula: str
    rationale: str
    node: AlphaNode
    metrics: BacktestMetrics
    fitness: float
    dsr: float = 0.0
    cpcv_mean_sharpe: float = 0.0
    cpcv_pct_positive: float = 0.0

def _collect_nodes(root: AlphaNode) -> list[AlphaNode]:
    nodes = [root]
    for c in root.children:
        nodes.extend(_collect_nodes(c))
    return nodes

def random_tree(rng: random.Random, max_depth: int = 3) -> AlphaNode:
    if max_depth <= 1 or rng.random() < 0.35:
        return AlphaNode("feature", rng.choice(["returns", "vwap", "close", "volume", "adv20", "high", "low"]))
    c = rng.random()
    if c < 0.30:
        return AlphaNode("op", rng.choice(UNARY_OPS), [random_tree(rng, max_depth - 1)])
    if c < 0.65:
        return AlphaNode("op", rng.choice(TS_UNARY_OPS), [random_tree(rng, max_depth - 1)], window=rng.choice(SAFE_WINDOWS))
    if c < 0.85:
        return AlphaNode("op", rng.choice(BINARY_OPS), [random_tree(rng, max_depth - 1), random_tree(rng, max_depth - 1)])
    return AlphaNode("op", "ts_corr", [random_tree(rng, max_depth - 1), random_tree(rng, max_depth - 1)], window=rng.choice([10, 20, 30]))

def mutate_tree(root: AlphaNode, rng: random.Random, max_depth: int = 7) -> AlphaNode:
    clone = root.clone()
    target = rng.choice(_collect_nodes(clone))
    if target.kind == "feature":
        target.value = rng.choice(sorted(ALLOWED_FEATURES))
    elif target.kind == "op":
        if target.window is not None and rng.random() < 0.6:
            target.window = rng.choice(SAFE_WINDOWS)
        elif target.value in UNARY_OPS:
            target.value = rng.choice(UNARY_OPS)
        elif target.value in TS_UNARY_OPS:
            target.value = rng.choice(TS_UNARY_OPS)
        elif target.value in BINARY_OPS:
            target.value = rng.choice(BINARY_OPS)
        else:
            sub = random_tree(rng, max_depth=2)
            target.kind, target.value, target.children, target.window = sub.kind, sub.value, sub.children, sub.window
    return root.clone() if clone.depth() > max_depth else clone

def crossover_trees(parent_a: AlphaNode, parent_b: AlphaNode, rng: random.Random, max_depth: int = 7) -> AlphaNode:
    child, donor = parent_a.clone(), parent_b.clone()
    t, s = rng.choice(_collect_nodes(child)), rng.choice(_collect_nodes(donor))
    t.kind, t.value, t.children, t.window = s.kind, s.value, s.children, s.window
    return parent_a.clone() if child.depth() > max_depth else child

class EvolutionaryAlphaEngine:
    def __init__(self, features: dict[str, np.ndarray], fwd_returns: np.ndarray, cost_bps: float = 3.0, max_corr_threshold: float = 0.50, seed: int = 42):
        self.features, self.fwd_returns = features, fwd_returns
        self.cost_bps, self.max_corr_threshold = cost_bps, max_corr_threshold
        self.rng = random.Random(seed)
        self.total_trials = 0
        self.alpha_zoo: list[AlphaCandidate] = []
        self._seen: set[str] = set()

    def _fitness(self, node: AlphaNode, m: BacktestMetrics) -> float:
        if np.isnan(m.sharpe_net) or m.sharpe_net <= 0.12 or abs(m.ic_tstat) < 1.1 or m.daily_turnover > 0.70:
            return -999.0
        return float(m.sharpe_net + 0.35 * abs(m.ic_tstat) - 0.02 * node.size() - 0.25 * m.daily_turnover)

    def _eval_parsed(self, node: AlphaNode, canon: str, rationale: str) -> AlphaCandidate | None:
        try:
            sig = node.evaluate(self.features)
            if not isinstance(sig, np.ndarray) or sig.shape != self.fwd_returns.shape or np.mean(np.isnan(sig)) > 0.50:
                return None
            m = evaluate_alpha_performance(sig, self.fwd_returns, cost_bps=self.cost_bps)
            return AlphaCandidate(canon, rationale, node, m, self._fitness(node, m))
        except Exception:
            return None

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[AlphaCandidate]:
        tasks = []
        for formula, rationale in items:
            try:
                node = parse_alpha(formula)
            except AlphaSyntaxError:
                continue
            canon = node.to_formula()
            if canon not in self._seen:
                self._seen.add(canon)
                self.total_trials += 1
                tasks.append((node, canon, rationale))
        if not tasks:
            return []
        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(lambda t: self._eval_parsed(*t), tasks))
        return [r for r in results if r is not None]

    def evaluate_candidate(self, formula: str, rationale: str = "Evolved") -> AlphaCandidate | None:
        res = self.evaluate_batch([(formula, rationale)])
        return res[0] if res else None

    def is_orthogonal(self, cand: AlphaCandidate) -> bool:
        for ex in self.alpha_zoo:
            corr = np.corrcoef(cand.metrics.net_returns, ex.metrics.net_returns)[0, 1]
            if not np.isnan(corr) and abs(corr) >= self.max_corr_threshold:
                return False
        return True

    def evolve(self, seed_hypotheses: list[tuple[str, str]], generations: int = 8, pop_size: int = 50, top_k_zoo: int = 5) -> list[AlphaCandidate]:
        pop = self.evaluate_batch(seed_hypotheses)
        while len(pop) < pop_size:
            batch = [(random_tree(self.rng, 3).to_formula(), "Random Seed") for _ in range(pop_size - len(pop) + 10)]
            pop.extend(self.evaluate_batch(batch))
        pop = pop[:pop_size]

        for _ in range(generations):
            pop.sort(key=lambda c: c.fitness, reverse=True)
            elites = pop[: max(8, pop_size // 4)]
            proposals = []
            for _ in range(pop_size * 2):
                if self.rng.random() < 0.65 and len(elites) >= 2:
                    p1, p2 = self.rng.sample(elites, 2)
                    proposals.append((crossover_trees(p1.node, p2.node, self.rng).to_formula(), f"Cross({p1.rationale[:12]}x{p2.rationale[:12]})"))
                else:
                    p = self.rng.choice(elites)
                    proposals.append((mutate_tree(p.node, self.rng).to_formula(), f"Mut({p.rationale[:18]})"))
            evaluated = self.evaluate_batch(proposals)
            pop = (elites + evaluated)[:pop_size]

        pop.sort(key=lambda c: c.fitness, reverse=True)
        for cand in pop:
            if len(self.alpha_zoo) >= top_k_zoo:
                break
            if cand.fitness > 0.0 and self.is_orthogonal(cand):
                sig = cand.node.evaluate(self.features)
                cand.dsr = deflated_sharpe_ratio(cand.metrics.net_returns, num_trials=max(1, self.total_trials))
                cpcv = run_cpcv_validation(sig, self.fwd_returns, cost_bps=self.cost_bps)
                cand.cpcv_mean_sharpe, cand.cpcv_pct_positive = cpcv["cpcv_mean_sharpe"], cpcv["cpcv_pct_positive"]
                self.alpha_zoo.append(cand)
        return self.alpha_zoo
