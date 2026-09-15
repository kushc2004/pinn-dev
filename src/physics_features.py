"""Deterministic thermodynamic feature map for symbolic discovery.

These are engineering proxies, not exact first-principles heat-transfer rates;
constant fluid-property factors are intentionally omitted because PySR can
absorb multiplicative constants while preserving the physically meaningful
structure.
"""

from __future__ import annotations

import numpy as np


NAMES = (
    "evap_delta_t",
    "evap_heat_proxy",
    "cond_delta_t",
    "cond_heat_proxy",
    "water_side_lift",
    "cond_refrigerant_pressure",
)


def numpy_features(x_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    evap_dt = x[:, 0] - x[:, 1]
    evap_q = x[:, 2] * evap_dt
    cond_dt = x[:, 5] - x[:, 4]
    cond_q = x[:, 6] * cond_dt
    water_lift = x[:, 4] - x[:, 1]
    pressure = x[:, 3]
    return np.column_stack((evap_dt, evap_q, cond_dt, cond_q, water_lift, pressure))


def torch_features(x_raw):
    import torch

    evap_dt = x_raw[:, 0] - x_raw[:, 1]
    evap_q = x_raw[:, 2] * evap_dt
    cond_dt = x_raw[:, 5] - x_raw[:, 4]
    cond_q = x_raw[:, 6] * cond_dt
    water_lift = x_raw[:, 4] - x_raw[:, 1]
    pressure = x_raw[:, 3]
    return torch.stack((evap_dt, evap_q, cond_dt, cond_q, water_lift, pressure), dim=1)
