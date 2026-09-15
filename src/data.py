"""Leakage-safe dataset splitting and preprocessing.

Two protocols are supported:

``random``
    IID interpolation benchmark using a deterministic 64/16/20
    train/validation/test split.

``high_load``
    Target-tail stress test: bottom 80% Motor KW -> train, 80--90th percentile
    -> validation, top 10% -> final test.

Every learned preprocessing statistic is fit on the training partition only.
The returned split keeps the original row indices so leakage assertions and
artifact audits can prove the partitions are disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass(frozen=True)
class Split:
    protocol: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_train_raw: np.ndarray
    x_val_raw: np.ndarray
    x_test_raw: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    metadata: dict

    def unscale_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH)
    required = set(config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Dataset contains missing values in required features/target")
    return df


def _random_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(n)
    n_test = int(round(n * config.RANDOM_TEST_FRACTION))
    test_idx = perm[:n_test]
    remainder = perm[n_test:]
    n_val = int(round(len(remainder) * config.RANDOM_VALIDATION_FRACTION_OF_REMAINDER))
    val_idx = remainder[:n_val]
    train_idx = remainder[n_val:]
    return train_idx, val_idx, test_idx, {
        "random_test_fraction": config.RANDOM_TEST_FRACTION,
        "random_validation_fraction_of_remainder": config.RANDOM_VALIDATION_FRACTION_OF_REMAINDER,
    }


def _high_load_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    q80 = float(np.quantile(y, config.HIGH_LOAD_TRAIN_QUANTILE))
    q90 = float(np.quantile(y, config.HIGH_LOAD_VALIDATION_QUANTILE))
    train_idx = np.where(y <= q80)[0]
    val_idx = np.where((y > q80) & (y <= q90))[0]
    test_idx = np.where(y > q90)[0]
    if min(len(train_idx), len(val_idx), len(test_idx)) < 500:
        raise ValueError(
            "High-load split produced a partition with <500 rows: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
    return train_idx, val_idx, test_idx, {
        "train_quantile": config.HIGH_LOAD_TRAIN_QUANTILE,
        "validation_quantile": config.HIGH_LOAD_VALIDATION_QUANTILE,
        "train_upper_kw": q80,
        "validation_upper_kw": q90,
    }


def _assert_disjoint(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, n: int) -> None:
    train, val, test = map(lambda a: set(map(int, a)), (train_idx, val_idx, test_idx))
    if train & val or train & test or val & test:
        raise AssertionError("Train/validation/test row indices overlap")
    if len(train | val | test) != n:
        raise AssertionError("Split does not cover every dataset row exactly once")


def make_split(protocol: str) -> Split:
    if protocol == "envelope":  # backward-compatible alias; new reports use high_load
        protocol = "high_load"
    if protocol not in {"random", "high_load"}:
        raise ValueError(f"Unknown protocol: {protocol}")

    df = load_dataset()
    X = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df[config.TARGET_COLUMN].to_numpy(dtype=np.float64)

    if protocol == "random":
        train_idx, val_idx, test_idx, metadata = _random_indices(len(df))
    else:
        train_idx, val_idx, test_idx, metadata = _high_load_indices(y)
    _assert_disjoint(train_idx, val_idx, test_idx, len(df))

    scaler_x = StandardScaler().fit(X[train_idx])
    scaler_y = StandardScaler().fit(y[train_idx].reshape(-1, 1))

    transform_x = lambda idx: scaler_x.transform(X[idx]).astype(np.float32)
    transform_y = lambda idx: scaler_y.transform(y[idx].reshape(-1, 1)).ravel().astype(np.float32)

    metadata = {
        **metadata,
        "protocol_version": config.PROTOCOL_VERSION,
        "n_total": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    return Split(
        protocol=protocol,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        x_train=transform_x(train_idx),
        y_train=transform_y(train_idx),
        x_val=transform_x(val_idx),
        y_val=transform_y(val_idx),
        x_test=transform_x(test_idx),
        y_test=transform_y(test_idx),
        x_train_raw=X[train_idx],
        x_val_raw=X[val_idx],
        x_test_raw=X[test_idx],
        y_train_raw=y[train_idx],
        y_val_raw=y[val_idx],
        y_test_raw=y[test_idx],
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        metadata=metadata,
    )


def split_audit(protocol: str) -> dict:
    """JSON-serializable proof of partition/scaler provenance."""
    split = make_split(protocol)
    return {
        "protocol": split.protocol,
        **split.metadata,
        "train_index_sha256": _array_sha256(split.train_idx),
        "validation_index_sha256": _array_sha256(split.val_idx),
        "test_index_sha256": _array_sha256(split.test_idx),
        "feature_scaler_mean": split.scaler_x.mean_.tolist(),
        "feature_scaler_scale": split.scaler_x.scale_.tolist(),
        "target_scaler_mean": split.scaler_y.mean_.tolist(),
        "target_scaler_scale": split.scaler_y.scale_.tolist(),
    }


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype=np.int64).tobytes()).hexdigest()


def data_sha256() -> str:
    digest = hashlib.sha256()
    with open(config.DATA_PATH, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
