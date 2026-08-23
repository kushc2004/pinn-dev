"""Shared configuration for the chiller PINN pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_chiller-3.csv"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"
EQUATION_PATH = RESULTS_DIR / "equation.json"
STATE_PATH = RESULTS_DIR / "state.json"

TARGET_COLUMN = "Motor KW"
FEATURE_COLUMNS = [
    "Evaporator Inlet Water Temperature",
    "Evaporator Outlet Water Temperature",
    "Evaporator Flow rate",
    "Condensor Refrigerant Pressure",
    "Condensor Inlet Water Temperature",
    "Condensor Outlet Water Temperature",
    "Condensor Flow Rate",
]

SEED = 42
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
HIDDEN_DIMS = (64, 32, 16)
CHECKPOINT_EVERY = 10

# Weight on the physics residual term for the PINN runs.
PINN_LAMBDA = 1.0

# Envelope protocol: train on samples below this quantile of Motor KW,
# evaluate both models on the held-out high-load region.
ENVELOPE_QUANTILE = 0.9

SR_SUBSAMPLE = 5000
SR_NITERATIONS = 40
