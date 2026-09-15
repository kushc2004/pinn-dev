"""Fast high-load test of deterministic physics-feature augmentation.

The comparison is parameter-count controlled: both models receive 13 inputs.
The control gets 7 standardized raw sensors plus 6 zero columns; the physics
model gets the same 7 sensors plus 6 train-standardized thermodynamic features.
Validation selects whether augmentation is worth testing. The final top-decile
test is untouched unless the physics model wins validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, data as data_mod, model as model_mod, physics_features
from .train import metrics


OUT = config.RESULTS_DIR / "fast_feature_aug_summary.json"
TUNING_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)


def _features(split):
    scaler = StandardScaler().fit(physics_features.numpy_features(split.x_train_raw))
    p_train = scaler.transform(physics_features.numpy_features(split.x_train_raw)).astype(np.float32)
    p_val = scaler.transform(physics_features.numpy_features(split.x_val_raw)).astype(np.float32)
    p_test = scaler.transform(physics_features.numpy_features(split.x_test_raw)).astype(np.float32)
    z_train = np.zeros_like(p_train)
    z_val = np.zeros_like(p_val)
    z_test = np.zeros_like(p_test)
    return {
        "control": (
            np.concatenate([split.x_train, z_train], axis=1),
            np.concatenate([split.x_val, z_val], axis=1),
            np.concatenate([split.x_test, z_test], axis=1),
        ),
        "physics_features": (
            np.concatenate([split.x_train, p_train], axis=1),
            np.concatenate([split.x_val, p_val], axis=1),
            np.concatenate([split.x_test, p_test], axis=1),
        ),
        "physics_scaler_mean": scaler.mean_.tolist(),
        "physics_scaler_scale": scaler.scale_.tolist(),
    }


def _fit(x_train, y_train, x_eval, split, seed: int):
    device = model_mod.pick_device()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = model_mod.build_model(n_features=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    mse = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(config.EPOCHS):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = net(xb).squeeze(-1)
            loss = mse(pred, yb)
            loss.backward()
            optimizer.step()
    net.eval()
    with torch.no_grad():
        pred_scaled = net(torch.from_numpy(x_eval).to(device)).squeeze(-1).cpu().numpy()
    return net, split.unscale_y(pred_scaled)


def _bootstrap(deltas: list[np.ndarray], n_boot: int = 5000):
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    values = np.empty(n_boot)
    for b in range(n_boot):
        seed_ids = rng.integers(0, len(deltas), size=len(deltas))
        means = []
        for sid in seed_ids:
            d = deltas[int(sid)]
            rows = rng.integers(0, len(d), size=len(d))
            means.append(float(np.mean(d[rows])))
        values[b] = float(np.mean(means))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> dict:
    split = data_mod.make_split("high_load")
    feats = _features(split)
    trained: dict[str, dict[int, torch.nn.Module]] = {"control": {}, "physics_features": {}}
    val_rows = []

    for kind in ("control", "physics_features"):
        x_train, x_val, _ = feats[kind]
        for seed in TUNING_SEEDS:
            net, pred = _fit(x_train, split.y_train, x_val, split, seed)
            trained[kind][seed] = net
            row = {"model": kind, "seed": seed, **metrics(split.y_val_raw, pred)}
            val_rows.append(row)
            print(json.dumps(row), flush=True)

    val_means = {
        kind: float(np.mean([r["mae"] for r in val_rows if r["model"] == kind]))
        for kind in ("control", "physics_features")
    }
    selected = min(val_means, key=val_means.get)
    summary = {
        "protocol_version": config.PROTOCOL_VERSION,
        "experiment": "high_load_parameter_controlled_physics_feature_augmentation",
        "physics_feature_names": list(physics_features.NAMES),
        "validation_mae_mean": val_means,
        "selected_on_validation": selected,
        "validation_runs": val_rows,
        "final_test_run": False,
    }

    if selected != "physics_features":
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(summary, indent=2) + "\n")
        print("Physics features did not win validation; final test remains untouched.", flush=True)
        return summary

    # Only after the architecture is frozen do we prepare the untouched test predictions.
    for kind in ("control", "physics_features"):
        x_train, _, _ = feats[kind]
        for seed in FINAL_SEEDS:
            if seed not in trained[kind]:
                net, _ = _fit(x_train, split.y_train, feats[kind][1], split, seed)
                trained[kind][seed] = net

    test_rows = []
    deltas = []
    control_mae, physics_mae = [], []
    device = model_mod.pick_device()
    for seed in FINAL_SEEDS:
        preds = {}
        for kind in ("control", "physics_features"):
            net = trained[kind][seed].to(device)
            net.eval()
            with torch.no_grad():
                p_scaled = net(torch.from_numpy(feats[kind][2]).to(device)).squeeze(-1).cpu().numpy()
            preds[kind] = split.unscale_y(p_scaled)
            row = {"model": kind, "seed": seed, **metrics(split.y_test_raw, preds[kind])}
            test_rows.append(row)
        control_mae.append(next(r["mae"] for r in test_rows if r["model"] == "control" and r["seed"] == seed))
        physics_mae.append(next(r["mae"] for r in test_rows if r["model"] == "physics_features" and r["seed"] == seed))
        deltas.append(np.abs(split.y_test_raw - preds["control"]) - np.abs(split.y_test_raw - preds["physics_features"]))

    mean_control = float(np.mean(control_mae))
    mean_physics = float(np.mean(physics_mae))
    ci = _bootstrap(deltas)
    summary.update(
        {
            "final_test_run": True,
            "final_seeds": list(FINAL_SEEDS),
            "test_runs": test_rows,
            "control_test_mae_mean": mean_control,
            "physics_feature_test_mae_mean": mean_physics,
            "mae_reduction_kw": mean_control - mean_physics,
            "mae_reduction_percent": (mean_control - mean_physics) / mean_control * 100.0,
            "paired_mae_improvement_95ci_kw": ci,
            "improvement_ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        }
    )
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
