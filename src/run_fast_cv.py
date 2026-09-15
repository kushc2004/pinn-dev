"""Repeated five-fold parameter-controlled CV for raw vs physics-feature MLPs."""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, data as data_mod, model as model_mod, physics_features
from .train import metrics


OUT = config.RESULTS_DIR / "repeated_five_fold_physics_feature_cv.json"
REPEAT_SEEDS = (42, 43, 44)
BOOTSTRAP_SAMPLES = 20000


def _fit(x_train, y_train, x_eval, y_scaler, seed):
    device = model_mod.pick_device()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = model_mod.build_model(n_features=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.MSELoss()
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
            opt.zero_grad()
            loss = loss_fn(net(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred_scaled = net(torch.from_numpy(x_eval).to(device)).squeeze(-1).cpu().numpy()
    return y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


def main():
    df = data_mod.load_dataset()
    x_raw = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = df[config.TARGET_COLUMN].to_numpy(dtype=np.float64)
    rows = []

    for repeat, split_seed in enumerate(REPEAT_SEEDS, start=1):
        kfold = KFold(n_splits=5, shuffle=True, random_state=split_seed)
        for fold, (train_idx, eval_idx) in enumerate(kfold.split(x_raw), start=1):
            sx = StandardScaler().fit(x_raw[train_idx])
            sy = StandardScaler().fit(y_raw[train_idx].reshape(-1, 1))
            sp = StandardScaler().fit(physics_features.numpy_features(x_raw[train_idx]))

            raw_train = sx.transform(x_raw[train_idx]).astype(np.float32)
            raw_eval = sx.transform(x_raw[eval_idx]).astype(np.float32)
            phys_train = sp.transform(physics_features.numpy_features(x_raw[train_idx])).astype(np.float32)
            phys_eval = sp.transform(physics_features.numpy_features(x_raw[eval_idx])).astype(np.float32)
            zeros_train = np.zeros_like(phys_train)
            zeros_eval = np.zeros_like(phys_eval)
            y_train = sy.transform(y_raw[train_idx].reshape(-1, 1)).ravel().astype(np.float32)

            variants = {
                "control": (
                    np.concatenate([raw_train, zeros_train], axis=1),
                    np.concatenate([raw_eval, zeros_eval], axis=1),
                ),
                "physics_features": (
                    np.concatenate([raw_train, phys_train], axis=1),
                    np.concatenate([raw_eval, phys_eval], axis=1),
                ),
            }
            model_seed = split_seed * 100 + fold
            for kind, (x_train, x_eval) in variants.items():
                pred = _fit(x_train, y_train, x_eval, sy, model_seed)
                row = {
                    "repeat": repeat,
                    "split_seed": split_seed,
                    "fold": fold,
                    "model_seed": model_seed,
                    "model": kind,
                    **metrics(y_raw[eval_idx], pred),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

    aggregate = {}
    for kind in ("control", "physics_features"):
        selected = [r for r in rows if r["model"] == kind]
        aggregate[kind] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in selected])),
                "std": float(np.std([r[metric] for r in selected], ddof=1)),
            }
            for metric in ("r2", "mae", "rmse", "mape")
        }
    base = aggregate["control"]["mae"]["mean"]
    phys = aggregate["physics_features"]["mae"]["mean"]
    paired = []
    for repeat in range(1, len(REPEAT_SEEDS) + 1):
        for fold in range(1, 6):
            c = next(r for r in rows if r["repeat"] == repeat and r["fold"] == fold and r["model"] == "control")
            p = next(r for r in rows if r["repeat"] == repeat and r["fold"] == fold and r["model"] == "physics_features")
            paired.append(c["mae"] - p["mae"])
    paired = np.asarray(paired, dtype=float)
    rng = np.random.default_rng(20260915)
    boot = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for i in range(BOOTSTRAP_SAMPLES):
        sample = rng.choice(paired, size=len(paired), replace=True)
        boot[i] = float(sample.mean())
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    result = {
        "experiment": "repeated_five_fold_parameter_controlled_physics_feature_cv",
        "n_rows": int(len(df)),
        "n_folds": 5,
        "n_repeats": len(REPEAT_SEEDS),
        "repeat_seeds": list(REPEAT_SEEDS),
        "n_paired_fold_results": int(len(paired)),
        "physics_feature_names": list(physics_features.NAMES),
        "folds": rows,
        "aggregate": aggregate,
        "mae_reduction_percent": (base - phys) / base * 100.0,
        "paired_mae_improvement_kw_mean": float(paired.mean()),
        "paired_mae_improvement_kw_std": float(paired.std(ddof=1)),
        "paired_mae_improvement_95ci_kw": ci,
        "improvement_ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "physics_feature_wins": int(np.sum(paired > 0)),
        "control_wins": int(np.sum(paired < 0)),
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
