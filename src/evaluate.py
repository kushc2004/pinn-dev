"""Aggregate metric fragments into results/metrics.csv and render figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config, model as model_mod, train as train_mod

METRIC_KEYS = ("r2", "mae", "rmse", "mape")


def collect_fragments() -> list[dict]:
    fragments = sorted(config.RESULTS_DIR.glob("metrics_*_*.json"))
    return [json.loads(path.read_text()) for path in fragments]


def write_metrics_csv(rows: list[dict]) -> Path:
    out = config.RESULTS_DIR / "metrics.csv"
    header = ["model", "protocol", *METRIC_KEYS, "lambda", "n_train", "n_test"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in header))
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    return out


def predict(model_kind: str, protocol: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the trained checkpoint and return (raw truth, raw prediction)."""
    from . import data as data_mod

    device = model_mod.pick_device()
    split = data_mod.make_split(protocol)
    ckpt_path = train_mod._checkpoint_path(model_kind, protocol)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint for {model_kind}/{protocol}: {ckpt_path}")
    net = model_mod.build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(split.x_test).to(device)).squeeze(-1).cpu().numpy()
    return split.y_test_raw, split.unscale_y(pred)


def parity_figure() -> Path:
    protocols = ["random", "envelope"]
    fig, axes = plt.subplots(1, len(protocols), figsize=(11, 5))
    colors = {"baseline": "#4878CF", "pinn": "#D65F5F"}
    for ax, protocol in zip(axes, protocols):
        for model_kind in ("baseline", "pinn"):
            y_true, y_pred = predict(model_kind, protocol)
            ax.scatter(y_true, y_pred, s=6, alpha=0.35, color=colors[model_kind], label=model_kind.upper())
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="0.3", linewidth=1)
        ax.set_title(f"{protocol} protocol")
        ax.set_xlabel("Actual Motor KW")
        ax.set_ylabel("Predicted Motor KW")
        ax.legend()
    fig.suptitle("Parity: baseline vs PINN")
    fig.tight_layout()
    out = config.FIGURES_DIR / "parity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def extrapolation_figure(rows: list[dict]) -> Path:
    envelope_rows = [row for row in rows if row.get("protocol") == "envelope"]
    if not envelope_rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(envelope_rows))
    labels = [f"{row['model'].upper()} (λ={row.get('lambda', '-')})" for row in envelope_rows]
    for ax, key, label in ((axes[0], "rmse", "RMSE (KW)"), (axes[1], "mape", "MAPE (%)")):
        ax.bar(x, [row[key] for row in envelope_rows], color=["#9AA6B2" if r["model"] == "baseline" else "#D65F5F" for r in envelope_rows])
        ax.set_xticks(x, labels)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Held-out high-load envelope")
    fig.tight_layout()
    out = config.FIGURES_DIR / "extrapolation_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def physics_fit_figure() -> Path | None:
    """Show how well each candidate power law tracks the held-out region alone."""
    if not config.EQUATION_PATH.is_file():
        return None
    from . import data as data_mod, physics

    device = model_mod.pick_device()
    split = data_mod.make_split("envelope")
    x_test = torch.from_numpy(split.x_test).to(device)

    curves = {}
    fn_discovered, expression = physics.physics_fn_from_equation()
    with torch.no_grad():
        curves[f"SR-discovered"] = fn_discovered(x_test).cpu().numpy()
    curves["legacy hard-coded"] = physics.legacy_physics_fn()(x_test).cpu().numpy()

    order = np.argsort(split.y_test)
    y_scaled_sorted = split.y_test[order]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(y_scaled_sorted, label="Actual Motor KW (scaled)", color="0.2", linewidth=1.5)
    for name, values in curves.items():
        ax.plot(values[order], alpha=0.7, label=name)
    ax.set_xlabel("Envelope test samples (sorted by load)")
    ax.set_ylabel("Scaled units")
    ax.set_title(f"Candidate power laws vs held-out high-load samples\n{expression}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = config.FIGURES_DIR / "physics_law_vs_envelope.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    rows = collect_fragments()
    if not rows:
        raise SystemExit("No metric fragments found; run training first")
    write_metrics_csv(rows)
    for path in (parity_figure(), extrapolation_figure(rows), physics_fit_figure()):
        if path:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
