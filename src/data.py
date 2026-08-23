"""Dataset loading and the two evaluation split protocols.

Both protocols share one rule: scalers are fitted on the training portion
only, so test metrics never see training statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class Split:
    """A train/test split in scaled space, with raw test values kept for reporting."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    y_test_raw: np.ndarray

    def unscale_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH)
    missing = set(config.FEATURE_COLUMNS + [config.TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def make_split(protocol: str) -> Split:
    """Build a Split for 'random' (80/20 iid) or 'envelope' (held-out high-load region)."""
    df = load_dataset()
    X = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df[config.TARGET_COLUMN].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(config.SEED)

    if protocol == "random":
        perm = rng.permutation(len(X))
        n_test = int(round(len(X) * config.TEST_FRACTION))
        test_idx, train_idx = perm[:n_test], perm[n_test:]
    elif protocol == "envelope":
        # Hold out the top decile of motor load as an out-of-envelope test
        # region; training only ever sees samples below the threshold.
        threshold = np.quantile(y, config.ENVELOPE_QUANTILE)
        test_idx = np.where(y > threshold)[0]
        train_idx = np.where(y <= threshold)[0]
        if len(test_idx) < 500:
            raise ValueError(
                f"Envelope hold-out too small ({len(test_idx)} rows); "
                "lower ENVELOPE_QUANTILE or condition on an input variable"
            )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    scaler_x = StandardScaler().fit(X[train_idx])
    scaler_y = StandardScaler().fit(y[train_idx].reshape(-1, 1))

    return Split(
        x_train=scaler_x.transform(X[train_idx]).astype(np.float32),
        y_train=scaler_y.transform(y[train_idx].reshape(-1, 1)).flatten().astype(np.float32),
        x_test=scaler_x.transform(X[test_idx]).astype(np.float32),
        y_test=scaler_y.transform(y[test_idx].reshape(-1, 1)).flatten().astype(np.float32),
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        y_test_raw=y[test_idx],
    )


def data_sha256() -> str:
    """Fingerprint of the dataset, used as a compatibility gate for checkpoints."""
    import hashlib

    digest = hashlib.sha256()
    with open(config.DATA_PATH, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
