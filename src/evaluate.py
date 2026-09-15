"""Final, test-only evaluation after protocol and lambda are frozen."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from . import config, data as data_mod, model as model_mod, physics, train as train_mod

METRIC_KEYS = ("r2", "mae", "rmse", "mape")


def _load_selection(protocol: str) -> dict:
    selection = json.loads(config.selection_path(protocol).read_text())
    if selection.get("protocol_version") != config.PROTOCOL_VERSION:
        raise RuntimeError(f"Stale lambda selection for {protocol}")
    return selection


def _predict_test(model_kind: str, protocol: str, seed: int, lam: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    split = data_mod.make_split(protocol)
    device = model_mod.pick_device()
    path = train_mod.checkpoint_path(model_kind, protocol, seed, lam)
    if not path.is_file():
        raise FileNotFoundError(f"Missing final checkpoint: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("protocol_version") != config.PROTOCOL_VERSION:
        raise RuntimeError(f"Checkpoint predates current protocol: {path}")
    net = model_mod.build_model().to(device)
    net.load_state_dict(payload["model_state"])
    net.eval()
    with torch.no_grad():
        pred_scaled = net(torch.from_numpy(split.x_test).to(device)).squeeze(-1).cpu().numpy()
    return split.y_test_raw, split.unscale_y(pred_scaled)


def _law_test_row(protocol: str) -> dict:
    split = data_mod.make_split(protocol)
    equation = json.loads(config.equation_path(protocol).read_text())
    fn = physics.compile_tree(equation["tree"])
    with torch.no_grad():
        pred_scaled = fn(torch.from_numpy(split.x_test)).cpu().numpy()
    pred_raw = split.unscale_y(pred_scaled)
    return {
        "model": "symbolic_surrogate",
        "protocol": protocol,
        "seed": "",
        "lambda": "",
        "n_test": len(split.test_idx),
        **train_mod.metrics(split.y_test_raw, pred_raw),
    }


def _fixed_tabular_baselines(protocol: str) -> list[dict]:
    """Untuned practical baselines; final test is used for reporting only."""
    split = data_mod.make_split(protocol)
    models = {
        "ridge": Ridge(alpha=1.0),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=config.SEED,
        ),
    }
    rows = []
    for name, estimator in models.items():
        estimator.fit(split.x_train, split.y_train_raw)
        pred = estimator.predict(split.x_test)
        rows.append(
            {
                "model": name,
                "protocol": protocol,
                "seed": config.SEED,
                "lambda": "",
                "n_test": len(split.test_idx),
                **train_mod.metrics(split.y_test_raw, pred),
            }
        )
    return rows


def _input_range_audit(protocol: str) -> dict:
    split = data_mod.make_split(protocol)
    features = {}
    for i, name in enumerate(config.FEATURE_COLUMNS):
        lo = float(split.x_train_raw[:, i].min())
        hi = float(split.x_train_raw[:, i].max())
        outside = (split.x_test_raw[:, i] < lo) | (split.x_test_raw[:, i] > hi)
        features[name] = {
            "train_min": lo,
            "train_max": hi,
            "test_min": float(split.x_test_raw[:, i].min()),
            "test_max": float(split.x_test_raw[:, i].max()),
            "test_fraction_outside_train_marginal_range": float(np.mean(outside)),
        }
    return features


def _hierarchical_bootstrap(deltas: list[np.ndarray]) -> tuple[float, float]:
    """95% CI for paired absolute-error improvement, resampling seeds then rows."""
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    values = np.empty(config.BOOTSTRAP_SAMPLES, dtype=float)
    n_seeds = len(deltas)
    for b in range(config.BOOTSTRAP_SAMPLES):
        sampled_seed_ids = rng.integers(0, n_seeds, size=n_seeds)
        seed_means = []
        for seed_id in sampled_seed_ids:
            d = deltas[int(seed_id)]
            sampled_rows = rng.integers(0, len(d), size=len(d))
            seed_means.append(float(np.mean(d[sampled_rows])))
        values[b] = float(np.mean(seed_means))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_protocol(protocol: str) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    selection = _load_selection(protocol)
    lam = float(selection["selected_lambda"])
    rows = []
    baseline_mae = []
    pinn_mae = []
    deltas = []
    baseline_predictions = []
    pinn_predictions = []
    y_ref = None

    for seed in config.FINAL_SEEDS:
        y, pred_base = _predict_test("baseline", protocol, seed)
        y2, pred_pinn = _predict_test("pinn", protocol, seed, lam)
        np.testing.assert_allclose(y, y2)
        y_ref = y
        base_metrics = train_mod.metrics(y, pred_base)
        pinn_metrics = train_mod.metrics(y, pred_pinn)
        rows.extend(
            [
                {"model": "baseline_mlp", "protocol": protocol, "seed": seed, "lambda": "", "n_test": len(y), **base_metrics},
                {"model": "physics_guided_mlp", "protocol": protocol, "seed": seed, "lambda": lam, "n_test": len(y), **pinn_metrics},
            ]
        )
        baseline_mae.append(base_metrics["mae"])
        pinn_mae.append(pinn_metrics["mae"])
        deltas.append(np.abs(y - pred_base) - np.abs(y - pred_pinn))
        baseline_predictions.append(pred_base)
        pinn_predictions.append(pred_pinn)

    rows.append(_law_test_row(protocol))
    rows.extend(_fixed_tabular_baselines(protocol))
    mean_base = float(np.mean(baseline_mae))
    mean_pinn = float(np.mean(pinn_mae))
    ci_low, ci_high = _hierarchical_bootstrap(deltas)
    summary = {
        "protocol": protocol,
        "protocol_version": config.PROTOCOL_VERSION,
        "selected_lambda": lam,
        "selection_metric": selection["selection_metric"],
        "final_seeds": list(config.FINAL_SEEDS),
        "baseline_test_mae_mean": mean_base,
        "baseline_test_mae_std": float(np.std(baseline_mae, ddof=1)),
        "pinn_test_mae_mean": mean_pinn,
        "pinn_test_mae_std": float(np.std(pinn_mae, ddof=1)),
        "mae_reduction_kw": mean_base - mean_pinn,
        "mae_reduction_percent": (mean_base - mean_pinn) / mean_base * 100.0,
        "paired_mae_improvement_95ci_kw": [ci_low, ci_high],
        "improvement_ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "input_range_audit": _input_range_audit(protocol),
    }
    predictions = {
        "y": np.asarray(y_ref),
        "baseline_mean": np.mean(np.stack(baseline_predictions), axis=0),
        "pinn_mean": np.mean(np.stack(pinn_predictions), axis=0),
    }
    return rows, summary, predictions


def _write_metrics(rows: list[dict]) -> Path:
    out = config.RESULTS_DIR / "metrics.csv"
    fields = ["model", "protocol", "seed", "lambda", "n_test", *METRIC_KEYS]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return out


def _write_summary(summaries: list[dict]) -> Path:
    out = config.RESULTS_DIR / "final_summary.json"
    out.write_text(json.dumps({"protocol_version": config.PROTOCOL_VERSION, "protocols": summaries}, indent=2) + "\n")
    return out


def _plot_parity(predictions: dict[str, dict[str, np.ndarray]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, protocol in zip(axes, ("random", "high_load")):
        data = predictions[protocol]
        ax.scatter(data["y"], data["baseline_mean"], s=7, alpha=0.3, label="Baseline MLP")
        ax.scatter(data["y"], data["pinn_mean"], s=7, alpha=0.3, label="Physics-guided MLP")
        lo = min(data["y"].min(), data["baseline_mean"].min(), data["pinn_mean"].min())
        hi = max(data["y"].max(), data["baseline_mean"].max(), data["pinn_mean"].max())
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
        ax.set_title(protocol.replace("_", " ").title())
        ax.set_xlabel("Actual Motor KW")
        ax.set_ylabel("Predicted Motor KW")
        ax.legend(fontsize=8)
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "parity_clean_protocol.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _plot_lambda_sweeps() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, protocol in zip(axes, ("random", "high_load")):
        selection = _load_selection(protocol)
        candidates = selection["candidates"]
        x = [row["lambda"] for row in candidates]
        y = [row["validation_mae_mean"] for row in candidates]
        err = [row["validation_mae_std"] for row in candidates]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3)
        ax.axvline(selection["selected_lambda"], linestyle="--", linewidth=1)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel("Physics weight λ")
        ax.set_ylabel("Validation MAE (kW)")
        ax.set_title(protocol.replace("_", " ").title())
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "lambda_validation_sweep.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> dict:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    predictions = {}
    for protocol in ("random", "high_load"):
        rows, summary, pred = evaluate_protocol(protocol)
        all_rows.extend(rows)
        summaries.append(summary)
        predictions[protocol] = pred
    metrics_path = _write_metrics(all_rows)
    summary_path = _write_summary(summaries)
    parity_path = _plot_parity(predictions)
    lambda_path = _plot_lambda_sweeps()
    result = {
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "figures": [str(parity_path), str(lambda_path)],
        "protocols": summaries,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
