"""Discover the chiller power law with symbolic regression.

Primary path is PySR (simulated-annealing genetic search over expressions).
If the Julia backend cannot bootstrap, falls back to gplearn. The winning
equation is stored as an op-tree in results/equation.json so train.py can
compile it into the PINN loss. Re-running skips discovery unless --force;
PySR's hall-of-fame directory under results/checkpoints/sr persists across
interrupted runs.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from . import config, data as data_mod, physics


def _subsample(split_seed: int = config.SEED):
    df = data_mod.load_dataset()
    X = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df[config.TARGET_COLUMN].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(split_seed)
    idx = rng.choice(len(X), size=min(config.SR_SUBSAMPLE, len(X)), replace=False)
    # Standardise jointly so the discovered law lives in the same scaled
    # space the networks train in.
    mu_x, sd_x = X.mean(axis=0), X.std(axis=0)
    mu_y, sd_y = y.mean(), y.std()
    return (
        (X[idx] - mu_x) / sd_x,
        ((y[idx] - mu_y) / sd_y),
    )


def _fit_pysr(X, y):
    from pysr import PySRRegressor

    regressor = PySRRegressor(
        niterations=config.SR_NITERATIONS,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        model_selection="best",
        progress=True,
        random_state=config.SEED,
        # Hall of fame streams here incrementally, so an interrupted
        # discovery still leaves its best-so-far equations on disk.
        output_directory=str(config.CHECKPOINT_DIR / "sr"),
        tempdir=str(config.CHECKPOINT_DIR / "sr-tmp"),
    )
    regressor.fit(X, y)
    best = regressor.get_best()
    return best.sympy_format, float(best.loss)


def _fit_gplearn(X, y):
    from gplearn.genetic import SymbolicRegressor

    regressor = SymbolicRegressor(
        function_set=("add", "sub", "mul", "div"),
        population_size=2000,
        generations=config.SR_NITERATIONS,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        parsimony_coefficient=0.001,
        random_state=config.SEED,
        n_jobs=-1,
        verbose=1,
    )
    regressor.fit(X, y)
    return regressor._program.__str__(), None


def _sympy_to_tree(expr) -> dict:
    import sympy

    if isinstance(expr, sympy.Symbol):
        return {"op": "var", "i": int(expr.name[1:])}
    if isinstance(expr, sympy.Float):
        return {"op": "const", "v": float(expr)}
    if isinstance(expr, sympy.Integer):
        return {"op": "const", "v": float(int(expr))}
    if isinstance(expr, sympy.Mul):
        args = [_sympy_to_tree(a) for a in expr.args]
        node = args[0]
        for arg in args[1:]:
            node = {"op": "mul", "a": node, "b": arg}
        return node
    if isinstance(expr, sympy.Add):
        args = [_sympy_to_tree(a) for a in expr.args]
        node = args[0]
        for arg in args[1:]:
            node = {"op": "add", "a": node, "b": arg}
        return node
    if isinstance(expr, sympy.Pow):
        base, exponent = (_sympy_to_tree(expr.base), _sympy_to_tree(expr.exp))
        if exponent == {"op": "const", "v": -1.0}:
            return {"op": "div", "a": {"op": "const", "v": 1.0}, "b": base}
        if exponent == {"op": "const", "v": 2.0}:
            return {"op": "square", "a": base}
        raise ValueError(f"Unsupported power: {expr}")
    raise ValueError(f"Unsupported sympy node: {type(expr)} {expr}")


_GPLEARN_TOKEN = re.compile(r"[(),]|[^\s(),]+")


def _gplearn_to_tree(program: str) -> dict:
    tokens = _GPLEARN_TOKEN.findall(program)
    pos = 0

    def parse() -> dict:
        nonlocal pos
        token = tokens[pos]
        pos += 1
        if token in {",", ")"}:
            raise ValueError(f"Unexpected token {token!r} in {program!r}")
        if token == "(":
            raise ValueError(f"Unexpected '(' in {program!r}")
        if tokens[pos : pos + 1] != ["("]:
            return _leaf(token)
        pos += 1  # consume '('
        args = []
        while tokens[pos] != ")":
            if tokens[pos] == ",":
                pos += 1
                continue
            args.append(parse())
        pos += 1  # consume ')'
        node: dict = {"op": token}
        for name, value in zip(("a", "b"), args):
            node[name] = value
        return node

    def _leaf(token: str) -> dict:
        match = re.fullmatch(r"X(\d+)", token)
        if match:
            return {"op": "var", "i": int(match.group(1))}
        return {"op": "const", "v": float(token)}

    return parse()


def _r2_against_target(tree: dict, X: np.ndarray, y: np.ndarray) -> float:
    import torch

    fn = physics.compile_tree(tree)
    with torch.no_grad():
        pred = fn(torch.from_numpy(X.astype(np.float32))).numpy()
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def discover(force: bool = False) -> dict:
    if config.EQUATION_PATH.is_file() and not force:
        equation = json.loads(config.EQUATION_PATH.read_text())
        print(f"Reusing existing {config.EQUATION_PATH} (pass --force to rediscover)")
        return equation

    X, y = _subsample()
    source, expression, loss = None, None, None
    try:
        expression, loss = _fit_pysr(X, y)
        tree = _sympy_to_tree(expression)
        source = "pysr"
    except Exception as error:  # noqa: BLE001 - any Julia bootstrap failure falls back
        print(f"PySR unavailable ({error}); falling back to gplearn")
        program, loss = _fit_gplearn(X, y)
        tree = _gplearn_to_tree(program)
        expression = program
        source = "gplearn"

    equation = {
        "source": source,
        "expression": str(expression),
        "tree": tree,
        "loss": loss,
        "n_rows": int(len(X)),
        "seed": config.SEED,
        "r2_vs_target": _r2_against_target(tree, X, y),
        "legacy_r2_vs_target": _r2_against_target(physics._legacy_tree(), X, y),
        "legacy_expression": physics.LEGACY_EXPRESSION,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EQUATION_PATH.write_text(json.dumps(equation, indent=2) + "\n")

    # Sanity check: the stored tree must round-trip through the torch compiler.
    assert np.isclose(_r2_against_target(tree, X, y), equation["r2_vs_target"])
    print(json.dumps({k: v for k, v in equation.items() if k != "tree"}, indent=2))
    return equation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rediscover even if equation.json exists")
    discover(parser.parse_args().force)
