"""Urgent high-load-only physics-feature PySR experiment.

Runs discovery and validation-only lambda selection first. If validation picks
lambda=0, the experiment stops without touching the final top-decile test. If a
positive lambda wins, it trains five paired final seeds and evaluates test once.
"""

from __future__ import annotations

import csv
import json

from . import config


def main() -> dict:
    from . import symbolic_regression

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    equation = symbolic_regression.discover("high_load", force=True)

    from . import train

    selection = train.tune_lambda("high_load", force=True)
    selected = float(selection["selected_lambda"])
    result = {
        "protocol_version": config.PROTOCOL_VERSION,
        "protocol": "high_load",
        "equation": {
            "source": equation["source"],
            "expression": equation["expression"],
            "train_discovery_r2": equation["train_discovery_r2"],
            "validation_r2": equation["validation_r2"],
            "input_feature_names": equation["input_feature_names"],
        },
        "selection": selection,
        "final_test_run": selected > 0.0,
    }

    if selected <= 0.0:
        path = config.RESULTS_DIR / "fast_high_load_summary.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print("Validation selected lambda=0; final top-decile test remains untouched.", flush=True)
        print(json.dumps(result, indent=2), flush=True)
        return result

    train.train_final("high_load", force=True)

    from . import evaluate

    rows, summary, _ = evaluate.evaluate_protocol("high_load")
    result["final_test"] = summary
    result_path = config.RESULTS_DIR / "fast_high_load_summary.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    metrics_path = config.RESULTS_DIR / "fast_high_load_metrics.csv"
    fields = ["model", "protocol", "seed", "lambda", "n_test", *evaluate.METRIC_KEYS]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
