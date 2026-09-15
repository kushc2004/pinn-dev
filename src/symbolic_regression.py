"""Protocol-specific, train-only symbolic-regression discovery.

The critical invariant is that PySR never receives validation/test rows or
their targets. ``data.make_split(protocol)`` fits all scalers on train only.
For the urgent high-load experiment, PySR receives train-only standardized
thermodynamic proxy features whose scaler provenance is stored with the
equation and reconstructed during neural training.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import config, data as data_mod, physics_features


def _subsample(protocol: str, split_seed: int = config.SEED):
    split = data_mod.make_split(protocol)
    rng = np.random.default_rng(split_seed)
    idx = rng.choice(
        len(split.x_train),
        size=min(config.SR_SUBSAMPLE, len(split.x_train)),
        replace=False,
    )
    if protocol == "high_load":
        from sklearn.preprocessing import StandardScaler

        physics_train_raw = physics_features.numpy_features(split.x_train_raw)
        physics_scaler = StandardScaler().fit(physics_train_raw)
        X = physics_scaler.transform(physics_train_raw[idx]).astype(np.float64)
        return split, X, split.y_train[idx].astype(np.float64), physics_scaler
    return split, split.x_train[idx].astype(np.float64), split.y_train[idx].astype(np.float64), None


def _fit_pysr_in_process(X, y, protocol: str):
    from pysr import PySRRegressor

    regressor = PySRRegressor(
        niterations=config.SR_NITERATIONS,
        populations=config.SR_POPULATIONS,
        ncycles_per_iteration=config.SR_CYCLES_PER_ITERATION,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        maxsize=config.SR_MAXSIZE,
        model_selection="best",
        progress=True,
        random_state=config.SEED,
        deterministic=False,
        parallelism="multithreading",
        batching=True,
        batch_size=config.SR_BATCH_SIZE,
        timeout_in_seconds=config.SR_TIMEOUT_SECONDS,
        input_stream="devnull",
        output_directory=str(config.CHECKPOINT_DIR / "sr" / protocol),
        tempdir=str(config.CHECKPOINT_DIR / "sr-tmp" / protocol),
    )
    regressor.fit(X, y)
    best = regressor.get_best()
    return best.sympy_format, float(best.loss)


def _fit_pysr(X, y, protocol: str):
    """Run PySR in a killable subprocess so a Julia wedge cannot stall the pipeline."""
    scratch_root = config.CHECKPOINT_DIR / "sr-subprocess" / protocol
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root) as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.npz"
        output_path = tmpdir / "output.json"
        np.savez_compressed(input_path, X=X, y=y)
        command = [
            sys.executable,
            "-m",
            "src.pysr_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--protocol",
            protocol,
        ]
        print(
            f"Launching PySR subprocess for {protocol} with "
            f"{config.SR_SUBPROCESS_TIMEOUT_SECONDS}s hard timeout",
            flush=True,
        )
        subprocess.run(
            command,
            cwd=config.ROOT,
            check=True,
            timeout=config.SR_SUBPROCESS_TIMEOUT_SECONDS,
        )
        payload = json.loads(output_path.read_text())
        import sympy

        return sympy.sympify(payload["expression"]), float(payload["loss"])


def _fit_gplearn(X, y):
    from gplearn.genetic import SymbolicRegressor

    regressor = SymbolicRegressor(
        function_set=("add", "sub", "mul", "div"),
        population_size=500,
        generations=min(5, config.SR_NITERATIONS),
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
        pos += 1
        args = []
        while tokens[pos] != ")":
            if tokens[pos] == ",":
                pos += 1
                continue
            args.append(parse())
        pos += 1
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


def _r2(tree: dict, X: np.ndarray, y: np.ndarray) -> float:
    import torch
    from . import physics

    fn = physics.compile_tree(tree)
    with torch.no_grad():
        pred = fn(torch.from_numpy(X.astype(np.float32))).cpu().numpy()
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def discover(protocol: str, force: bool = False) -> dict:
    if protocol == "envelope":
        protocol = "high_load"
    path = config.equation_path(protocol)
    if path.is_file() and not force:
        equation = json.loads(path.read_text())
        if equation.get("protocol_version") == config.PROTOCOL_VERSION:
            print(f"Reusing existing {path} (pass --force to rediscover)")
            return equation

    split, X, y, physics_scaler = _subsample(protocol)
    source, expression, loss = None, None, None
    try:
        expression, loss = _fit_pysr(X, y, protocol)
        tree = _sympy_to_tree(expression)
        source = "pysr"
    except Exception as error:  # any Julia bootstrap failure falls back
        print(f"PySR unavailable ({error}); falling back to gplearn")
        program, loss = _fit_gplearn(X, y)
        tree = _gplearn_to_tree(program)
        expression = program
        source = "gplearn"

    audit = data_mod.split_audit(protocol)
    if physics_scaler is not None:
        X_val_symbolic = physics_scaler.transform(
            physics_features.numpy_features(split.x_val_raw)
        ).astype(np.float64)
    else:
        X_val_symbolic = split.x_val.astype(np.float64)
    equation = {
        "protocol_version": config.PROTOCOL_VERSION,
        "protocol": protocol,
        "source": source,
        "expression": str(expression),
        "tree": tree,
        "loss": loss,
        "input_space": "physics_features" if physics_scaler is not None else "standardized_raw_features",
        "input_feature_names": list(physics_features.NAMES) if physics_scaler is not None else list(config.FEATURE_COLUMNS),
        "n_discovery_rows": int(len(X)),
        "seed": config.SEED,
        "search_config": {
            "niterations": config.SR_NITERATIONS,
            "populations": config.SR_POPULATIONS,
            "ncycles_per_iteration": config.SR_CYCLES_PER_ITERATION,
            "batching": True,
            "batch_size": config.SR_BATCH_SIZE,
            "maxsize": config.SR_MAXSIZE,
            "parallelism": "multithreading",
            "timeout_in_seconds": config.SR_TIMEOUT_SECONDS,
            "subprocess_timeout_in_seconds": config.SR_SUBPROCESS_TIMEOUT_SECONDS,
        },
        "train_discovery_r2": _r2(tree, X, y),
        "validation_r2": _r2(tree, X_val_symbolic, split.y_val),
        "train_index_sha256": audit["train_index_sha256"],
        "feature_scaler_mean": audit["feature_scaler_mean"],
        "feature_scaler_scale": audit["feature_scaler_scale"],
        "target_scaler_mean": audit["target_scaler_mean"],
        "target_scaler_scale": audit["target_scaler_scale"],
    }
    if physics_scaler is not None:
        equation.update(
            {
                "raw_feature_scaler_mean": audit["feature_scaler_mean"],
                "raw_feature_scaler_scale": audit["feature_scaler_scale"],
                "physics_feature_scaler_mean": physics_scaler.mean_.tolist(),
                "physics_feature_scaler_scale": physics_scaler.scale_.tolist(),
            }
        )
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(equation, indent=2) + "\n")
    assert np.isclose(_r2(tree, X, y), equation["train_discovery_r2"])
    print(json.dumps({k: v for k, v in equation.items() if k != "tree"}, indent=2))
    return equation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=("random", "high_load"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    discover(args.protocol, args.force)
