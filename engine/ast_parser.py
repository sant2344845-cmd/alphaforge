from __future__ import annotations
import ast, copy
from dataclasses import dataclass, field
import numpy as np
from engine.operators import OPERATOR_REGISTRY

ALLOWED_FEATURES = {"open", "high", "low", "close", "volume", "returns", "vwap", "adv20"}

class AlphaSyntaxError(ValueError):
    pass

@dataclass
class AlphaNode:
    kind: str
    value: str | float
    children: list[AlphaNode] = field(default_factory=list)
    window: int | None = None

    def evaluate(self, features: dict[str, np.ndarray]) -> np.ndarray:
        if self.kind == "feature":
            return features[str(self.value)]
        if self.kind == "const":
            return float(self.value)  # type: ignore[return-value]
        if self.kind == "op":
            fn = OPERATOR_REGISTRY[str(self.value)]["fn"]
            args = [c.evaluate(features) for c in self.children]
            return fn(*args, self.window) if self.window is not None else fn(*args)
        raise AlphaSyntaxError(f"Unknown node: {self.kind}")

    def to_formula(self) -> str:
        if self.kind == "feature": return str(self.value)
        if self.kind == "const": return f"{self.value:g}"
        args = [c.to_formula() for c in self.children] + ([str(self.window)] if self.window is not None else [])
        return f"{self.value}({', '.join(args)})"

    def depth(self) -> int:
        return 1 if not self.children else 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def clone(self) -> AlphaNode:
        return copy.deepcopy(self)

def _convert_py_ast(node: ast.AST, max_window: int = 252) -> AlphaNode:
    if isinstance(node, ast.Expression):
        return _convert_py_ast(node.body, max_window)
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_FEATURES:
            raise AlphaSyntaxError(f"Variabile non ammessa: {node.id}")
        return AlphaNode("feature", node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return AlphaNode("const", float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return AlphaNode("op", "neg", [_convert_py_ast(node.operand, max_window)])
    if isinstance(node, ast.BinOp):
        op_map = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
        if type(node.op) not in op_map:
            raise AlphaSyntaxError("Operatore binario non supportato")
        return AlphaNode("op", op_map[type(node.op)], [_convert_py_ast(node.left, max_window), _convert_py_ast(node.right, max_window)])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in OPERATOR_REGISTRY:
        op_name = node.func.id
        spec = OPERATOR_REGISTRY[op_name]
        arity, has_w = spec["arity"], "d" in spec["params"]
        if len(node.args) != arity + (1 if has_w else 0) or node.keywords:
            raise AlphaSyntaxError(f"Argomenti errati per {op_name}")
        children = [_convert_py_ast(a, max_window) for a in node.args[:arity]]
        win_val = None
        if has_w:
            w = node.args[-1]
            if not (isinstance(w, ast.Constant) and isinstance(w.value, int) and 1 <= int(w.value) <= max_window):
                raise AlphaSyntaxError(f"Finestra non valida per {op_name}")
            win_val = int(w.value)
        return AlphaNode("op", op_name, children, win_val)
    raise AlphaSyntaxError("Costrutto non permesso")

def parse_alpha(formula: str, max_depth: int = 8, max_window: int = 252) -> AlphaNode:
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise AlphaSyntaxError(str(exc)) from exc
    root = _convert_py_ast(tree, max_window)
    if root.depth() > max_depth:
        raise AlphaSyntaxError("Formula troppo profonda")
    return root
