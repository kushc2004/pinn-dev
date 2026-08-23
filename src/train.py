"""Train the baseline or physics-informed model on one split protocol.

The PINN loss is computed per batch: MSE(y, y_hat) + lambda * MSE(physics(x),
y_hat). Unlike the legacy TF scripts, the physics residual uses the batch's
own features, so it constrains each prediction individually. The residual is
standardised with statistics from the training set so lambda is comparable to
the data-loss weight.

Checkpoints are written every CHECKPOINT_EVERY epochs and support --resume.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, data as data_mod, model as model_mod, physics


def _checkpoint_path(model_name: str, protocol: str) -> Path:
    return config.CHECKPOINT_DIR / f"{model_name}_{protocol}_latest.pt"


def _metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict:
    diff = y_true_raw - y_pred_raw
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true_raw - y_true_raw.mean()) ** 2))
    nonzero = y_true_raw != 0
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mape": float(np.mean(np.abs(diff[nonzero] / y_true_raw[nonzero])) * 100),
    }


def train(
    model_kind: str,
    protocol: str,
    lam: float = config.PINN_LAMBDA,
    resume: bool = False,
) -> dict:
    if model_kind not in {"baseline", "pinn"}:
        raise ValueError(f"Unknown model kind: {model_kind}")
    device = model_mod.pick_device()
    torch.manual_seed(config.SEED)

    split = data_mod.make_split(protocol)
    x_train = torch.from_numpy(split.x_train)
    y_train = torch.from_numpy(split.y_train)
    x_test = torch.from_numpy(split.x_test)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.SEED),
    )

    net = model_mod.build_model().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    physics_fn, standardise = None, (0.0, 1.0)
    if model_kind == "pinn":
        physics_fn, _ = physics.physics_fn_from_equation()
        with torch.no_grad():
            residuals = physics_fn(x_train).numpy()
        standardise = (float(residuals.mean()), float(residuals.std()))

    start_epoch = 0
    ckpt_path = _checkpoint_path(model_kind, protocol)
    if resume and ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("data_sha256") != data_mod.data_sha256():
            print("Checkpoint predates a dataset change; starting fresh")
        else:
            net.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            standardise = ckpt["physics_standardisation"]
            print(f"Resumed {model_kind}/{protocol} at epoch {start_epoch}")

    mse_loss = nn.MSELoss()
    for epoch in range(start_epoch, config.EPOCHS):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = net(xb).squeeze(-1)
            loss = mse_loss(yb, pred)
            if physics_fn is not None:
                mu, sd = standardise
                residual = (physics_fn(xb) - mu) / max(sd, 1e-8)
                loss = loss + lam * mse_loss(residual, pred)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % config.CHECKPOINT_EVERY == 0 or epoch + 1 == config.EPOCHS:
            _save_checkpoint(
                ckpt_path, net, optimizer, epoch, standardise, model_kind, protocol
            )
            print(f"[{model_kind}/{protocol}] epoch {epoch + 1}/{config.EPOCHS} loss {loss.item():.4f}", flush=True)

    net.eval()
    with torch.no_grad():
        pred_test = net(x_test.to(device)).squeeze(-1).cpu().numpy()
    metrics = _metrics(split.y_test_raw, split.unscale_y(pred_test))

    result = {
        "model": model_kind,
        "protocol": protocol,
        "lambda": lam,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        **metrics,
    }
    fragment = config.RESULTS_DIR / f"metrics_{model_kind}_{protocol}.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fragment.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def _save_checkpoint(path, net, optimizer, epoch, standardise, model_kind, protocol):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "physics_standardisation": list(standardise),
        "seed": config.SEED,
        "data_sha256": data_mod.data_sha256(),
        "hidden_dims": list(config.HIDDEN_DIMS),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    shutil.move(tmp, path)
