"""Shared configuration for the chiller physics-guided regression pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_chiller-3.csv"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"
STATE_PATH = RESULTS_DIR / "state.json"

# Increment whenever an evaluation, training, or search-semantic change should invalidate cached stages.
PROTOCOL_VERSION = "clean-v5-physics-feature-pysr-highload-20260915"

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
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
HIDDEN_DIMS = (64, 32, 16)
CHECKPOINT_EVERY = 10

# Random interpolation protocol: 20% test, then 20% of the remaining 80% for
# validation -> 64/16/20 train/validation/test overall.
RANDOM_TEST_FRACTION = 0.20
RANDOM_VALIDATION_FRACTION_OF_REMAINDER = 0.20

# High-load target-tail protocol. These boundaries are computed from Motor KW
# once, before any learned preprocessing. The final top decile is never used
# for scaling, PySR discovery, lambda selection, or early model decisions.
HIGH_LOAD_TRAIN_QUANTILE = 0.80
HIGH_LOAD_VALIDATION_QUANTILE = 0.90

# Physics-weight selection is validation-only. Test metrics are produced only
# after this sweep has selected a frozen lambda.
LAMBDA_CANDIDATES = (0.00, 0.02, 0.05, 0.10, 0.20)
TUNING_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)

# Symbolic regression is fit only on the main training split for each protocol.
SR_SUBSAMPLE = 5000
SR_NITERATIONS = 10
SR_POPULATIONS = 4
SR_CYCLES_PER_ITERATION = 150
SR_BATCH_SIZE = 512
SR_MAXSIZE = 20
SR_TIMEOUT_SECONDS = 180
SR_SUBPROCESS_TIMEOUT_SECONDS = 240

# Hierarchical bootstrap of paired absolute-error differences across seeds and
# rows, used for the final test-set improvement confidence interval.
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260915


def equation_path(protocol: str) -> Path:
    return RESULTS_DIR / f"equation_{protocol}.json"


def selection_path(protocol: str) -> Path:
    return RESULTS_DIR / f"lambda_selection_{protocol}.json"
