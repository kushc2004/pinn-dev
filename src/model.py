"""Shared MLP architecture for baseline and PINN runs.

Identical architecture and identical seed-42 initialisation guarantee the
baseline/PINN comparison starts from the same weights.
"""

from __future__ import annotations

import torch
from torch import nn

from . import config


def build_model(n_features: int = len(config.FEATURE_COLUMNS)) -> nn.Module:
    layers: list[nn.Module] = []
    in_dim = n_features
    for dim in config.HIDDEN_DIMS:
        layers += [nn.Linear(in_dim, dim), nn.ReLU()]
        in_dim = dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
