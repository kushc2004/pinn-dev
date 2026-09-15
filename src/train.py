"""Train leakage-safe baseline/PINN candidates and tune lambda on validation.

No function in this module reads ``split.x_test`` or ``split.y_test``. Final
test evaluation lives in :mod:`src.evaluate` and runs only after a lambda has
been frozen in ``results/lambda_selection_<protocol>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, data as data_mod, model as model_mod, physics


def checkpoint_path(model_name: str, protocol: str, seed: int, lam: float | None = None) -> Path:
    if protocol == "envelope":
        protocol = "high_load"
    suffix = "" if model_name == "baseline" else f"_lam{lam:g}"
    return config.CHECKPOINT_DIR / protocol / f"{model_name}_seed{seed}{suffix}.pt"


def validation_fragment_path(model_name: str, protocol: str, seed: int, lam: float | None = None) -> Path:
    suffix = "" if model_name == "baseline" else f"_lam{lam:g}"
    return config.RESULTS_DIR / "validation" / protocol / f"{model_name}_seed{seed}{suffix}.json"


def metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict:
    diff = np.asarray(y_true_raw) - np.asarray(y_pred_raw)
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((np.asarray(y_true_raw) - np.mean(y_true_raw)) ** 2))
    nonzero = np.asarray(y_true_raw) != 0
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mape": float(np.mean(np.abs(diff[nonzero] / np.asarray(y_true_raw)[nonzero])) * 100),
    }


def _indices_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _save_checkpoint(path: Path, net, optimizer, epoch: int, *, model_kind: str, protocol: str, seed: int, lam: float | None, train_idx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "model_kind": model_kind,
        "protocol": protocol,
        "lambda": lam,
        "seed": seed,
        "protocol_version": config.PROTOCOL_VERSION,
        "data_sha256": data_mod.data_sha256(),
        "train_index_sha256": _indices_hash(train_idx),
        "hidden_dims": list(config.HIDDEN_DIMS),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    shutil.move(tmp, path)


def fit_candidate(
    model_kind: str,
    protocol: str,
    seed: int,
    lam: float | None = None,
    resume: bool = True,
) -> dict:
    if protocol == "envelope":
        protocol = "high_load"
    if model_kind not in {"baseline", "pinn"}:
        raise ValueError(f"Unknown model kind: {model_kind}")
    if model_kind == "pinn" and lam is None:
        raise ValueError("PINN candidate requires lambda")

    split = data_mod.make_split(protocol)
    device = model_mod.pick_device()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_train = torch.from_numpy(split.x_train)
    y_train = torch.from_numpy(split.y_train)
    x_val = torch.from_numpy(split.x_val)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    net = model_mod.build_model().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    physics_fn = None
    if model_kind == "pinn" and float(lam) != 0.0:
        physics_fn, _ = physics.physics_fn_from_equation(protocol)

    ckpt = checkpoint_path(model_kind, protocol, seed, lam)
    start_epoch = 0
    if resume and ckpt.is_file():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        compatible = (
            payload.get("data_sha256") == data_mod.data_sha256()
            and payload.get("protocol_version") == config.PROTOCOL_VERSION
            and payload.get("train_index_sha256") == _indices_hash(split.train_idx)
            and payload.get("seed") == seed
            and payload.get("lambda") == lam
        )
        if compatible:
            net.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            start_epoch = int(payload["epoch"]) + 1
            print(f"Resumed {model_kind}/{protocol}/seed={seed}/lambda={lam} at epoch {start_epoch}")

    mse = nn.MSELoss()
    for epoch in range(start_epoch, config.EPOCHS):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = net(xb).squeeze(-1)
            loss = mse(yb, pred)
            if physics_fn is not None:
                # PySR was fit to the same train-standardized X and y, so its
                # output already lives in the network target coordinates.
                physics_target = physics_fn(xb)
                loss = loss + float(lam) * mse(physics_target, pred)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % config.CHECKPOINT_EVERY == 0 or epoch + 1 == config.EPOCHS:
            _save_checkpoint(
                ckpt,
                net,
                optimizer,
                epoch,
                model_kind=model_kind,
                protocol=protocol,
                seed=seed,
                lam=lam,
                train_idx=split.train_idx,
            )

    net.eval()
    with torch.no_grad():
        val_scaled = net(x_val.to(device)).squeeze(-1).cpu().numpy()
    result = {
        "protocol_version": config.PROTOCOL_VERSION,
        "model": model_kind,
        "protocol": protocol,
        "seed": seed,
        "lambda": lam,
        "n_train": len(split.train_idx),
        "n_validation": len(split.val_idx),
        **metrics(split.y_val_raw, split.unscale_y(val_scaled)),
    }
    fragment = validation_fragment_path(model_kind, protocol, seed, lam)
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def tune_lambda(protocol: str, force: bool = False) -> dict:
    if protocol == "envelope":
        protocol = "high_load"
    if not config.equation_path(protocol).is_file():
        raise FileNotFoundError(f"Run symbolic regression first: {config.equation_path(protocol)}")

    candidates = []
    for lam in config.LAMBDA_CANDIDATES:
        runs = [fit_candidate("pinn", protocol, seed, lam, resume=not force) for seed in config.TUNING_SEEDS]
        candidates.append(
            {
                "lambda": lam,
                "validation_mae_mean": float(np.mean([r["mae"] for r in runs])),
                "validation_mae_std": float(np.std([r["mae"] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                "runs": runs,
            }
        )
    best = min(candidates, key=lambda row: (row["validation_mae_mean"], row["lambda"]))
    selection = {
        "protocol_version": config.PROTOCOL_VERSION,
        "protocol": protocol,
        "selection_metric": "validation_mae_mean",
        "tuning_seeds": list(config.TUNING_SEEDS),
        "selected_lambda": best["lambda"],
        "candidates": candidates,
    }
    path = config.selection_path(protocol)
    path.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected lambda={best['lambda']} for {protocol} using validation only")
    return selection


def train_final(protocol: str, force: bool = False) -> dict:
    if protocol == "envelope":
        protocol = "high_load"
    selection = json.loads(config.selection_path(protocol).read_text())
    if selection.get("protocol_version") != config.PROTOCOL_VERSION:
        raise RuntimeError("Lambda selection predates the current protocol")
    lam = float(selection["selected_lambda"])
    records = []
    for seed in config.FINAL_SEEDS:
        records.append(fit_candidate("baseline", protocol, seed, resume=not force))
        records.append(fit_candidate("pinn", protocol, seed, lam=lam, resume=not force))
    return {"protocol": protocol, "selected_lambda": lam, "validation_records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tune = sub.add_parser("tune")
    tune.add_argument("--protocol", required=True, choices=("random", "high_load"))
    tune.add_argument("--force", action="store_true")
    final = sub.add_parser("final")
    final.add_argument("--protocol", required=True, choices=("random", "high_load"))
    final.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "tune":
        tune_lambda(args.protocol, args.force)
    else:
        train_final(args.protocol, args.force)
