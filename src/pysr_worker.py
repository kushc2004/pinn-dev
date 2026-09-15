"""Isolated PySR worker used so the parent process can enforce a hard timeout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .symbolic_regression import _fit_pysr_in_process


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True, choices=("random", "high_load"))
    args = parser.parse_args()

    arrays = np.load(args.input)
    expression, loss = _fit_pysr_in_process(arrays["X"], arrays["y"], args.protocol)
    Path(args.output).write_text(
        json.dumps({"expression": str(expression), "loss": float(loss)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
