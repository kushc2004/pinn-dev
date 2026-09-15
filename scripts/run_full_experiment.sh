#!/bin/sh
# Run the clean protocol: train-only SR -> validation lambda sweep ->
# multi-seed final models -> one-shot final test evaluation.
# Pass --force to ignore completed-stage checkpoints.
set -e
cd "$(dirname "$0")/.."
exec python -m src.run_all "$@"
