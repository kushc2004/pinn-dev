#!/bin/sh
# Run the full experiment pipeline (SR discovery -> 4 trainings -> evaluate).
# Pass --force to ignore completed-stage checkpoints.
set -e
cd "$(dirname "$0")/.."
exec python -m src.run_all "$@"
