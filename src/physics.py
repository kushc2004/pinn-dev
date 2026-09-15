"""Physics residual: the SR-discovered chiller power law, evaluated in torch.

The discovered equation is stored in ``results/equation.json`` as an op-tree
(robust to round-tripping from PySR/gplearn) plus a human-readable string.
The tree is compiled into a torch function so the physics residual runs
batch-wise on whatever device the model trains on. Division is guarded so a
denominator near zero cannot poison the loss.
"""

from __future__ import annotations

import json
from typing import Callable

import torch

from . import config
from . import physics_features

def _c(value: float) -> dict:
    return {"op": "const", "v": value}


def _var(index: int) -> dict:
    return {"op": "var", "i": index}


# The hardcoded power law recovered from the legacy TF scripts, kept for
# comparison against the rediscovered equation. Written against the semantic
# feature roles (x0 evaporator inlet temp ... x5 condenser outlet temp,
# x6 condenser flow rate); note the legacy script's own column indexing was
# inconsistent, so this encodes its intent rather than its exact arithmetic.
LEGACY_EXPRESSION = "(x5*x0 + (x0*0.548 - (x4 - (x2 + (x6*x4*-0.125)/x5)))) * ((x5 + 0.490) * (0.455/x5))"


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    signed_eps = torch.where(b < 0, torch.full_like(b, -eps), torch.full_like(b, eps))
    return a / torch.where(b.abs() < eps, signed_eps, b)


def _eval_node(node: dict, columns: dict[str, torch.Tensor]) -> torch.Tensor:
    op = node["op"]
    if op == "var":
        return columns[f"x{node['i']}"]
    if op == "const":
        return torch.full_like(columns["x0"], float(node["v"]))
    if op == "add":
        return _eval_node(node["a"], columns) + _eval_node(node["b"], columns)
    if op == "sub":
        return _eval_node(node["a"], columns) - _eval_node(node["b"], columns)
    if op == "mul":
        return _eval_node(node["a"], columns) * _eval_node(node["b"], columns)
    if op == "div":
        return _safe_div(_eval_node(node["a"], columns), _eval_node(node["b"], columns))
    if op == "neg":
        return -_eval_node(node["a"], columns)
    unary = {
        "abs": torch.abs,
        "square": lambda t: t * t,
        "sqrt": lambda t: torch.sqrt(torch.clamp(t, min=0.0)),
        "log": lambda t: torch.log(torch.clamp(t, min=1e-6)),
        "exp": torch.exp,
        "sin": torch.sin,
        "cos": torch.cos,
    }
    if op in unary:
        return unary[op](_eval_node(node["a"], columns))
    raise ValueError(f"Unknown op in equation tree: {op}")


def compile_tree(tree: dict) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compile an op-tree into f(x) with x of shape (batch, n_features)."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        columns = {f"x{i}": x[:, i] for i in range(x.shape[1])}
        return _eval_node(tree, columns)

    return fn


def load_equation(protocol: str) -> dict:
    if protocol == "envelope":
        protocol = "high_load"
    with open(config.equation_path(protocol)) as handle:
        return json.load(handle)


def physics_fn_from_equation(protocol: str) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    """Return (compiled physics function, human-readable expression)."""
    equation = load_equation(protocol)
    base = compile_tree(equation["tree"])
    if equation.get("input_space") != "physics_features":
        return base, equation["expression"]

    x_mean = torch.tensor(equation["raw_feature_scaler_mean"], dtype=torch.float32)
    x_scale = torch.tensor(equation["raw_feature_scaler_scale"], dtype=torch.float32)
    p_mean = torch.tensor(equation["physics_feature_scaler_mean"], dtype=torch.float32)
    p_scale = torch.tensor(equation["physics_feature_scaler_scale"], dtype=torch.float32)

    def fn(x_scaled: torch.Tensor) -> torch.Tensor:
        xm = x_mean.to(device=x_scaled.device, dtype=x_scaled.dtype)
        xs = x_scale.to(device=x_scaled.device, dtype=x_scaled.dtype)
        pm = p_mean.to(device=x_scaled.device, dtype=x_scaled.dtype)
        ps = p_scale.to(device=x_scaled.device, dtype=x_scaled.dtype)
        x_raw = x_scaled * xs + xm
        p_raw = physics_features.torch_features(x_raw)
        p_scaled = (p_raw - pm) / ps
        return base(p_scaled)

    return fn, equation["expression"]


def legacy_physics_fn() -> Callable[[torch.Tensor], torch.Tensor]:
    return compile_tree(_legacy_tree())


def _legacy_tree() -> dict:
    flow_term = {
        "op": "div",
        "a": {"op": "mul", "a": {"op": "mul", "a": _var(6), "b": _var(4)}, "b": _c(-0.125)},
        "b": _var(5),
    }
    lift_term = {"op": "sub", "a": _var(4), "b": {"op": "add", "a": _var(2), "b": flow_term}}
    first_factor = {
        "op": "add",
        "a": {"op": "mul", "a": _var(5), "b": _var(0)},
        "b": {"op": "sub", "a": {"op": "mul", "a": _var(0), "b": _c(0.548)}, "b": lift_term},
    }
    second_factor = {
        "op": "mul",
        "a": {"op": "add", "a": _var(5), "b": _c(0.490)},
        "b": {"op": "div", "a": _c(0.455), "b": _var(5)},
    }
    return {"op": "mul", "a": first_factor, "b": second_factor}
