"""Run the full experiment pipeline, skipping stages already recorded complete.

State lives in results/state.json and gates both local reruns and Kaggle
resume: a stage runs only if its recorded status is not 'complete' or the
dataset fingerprint changed underneath it.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from . import config, data as data_mod


def _load_state() -> dict:
    if config.STATE_PATH.is_file():
        return json.loads(config.STATE_PATH.read_text())
    return {"stages": {}}


def _save_state(state: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state["data_sha256"] = data_mod.data_sha256()
    try:
        state["git_commit"] = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        state["git_commit"] = None
    config.STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def _stage_complete(state: dict, name: str) -> bool:
    stage = state["stages"].get(name)
    if not stage or stage.get("status") != "complete":
        return False
    # A dataset change invalidates every recorded stage.
    if state.get("data_sha256") != data_mod.data_sha256():
        return False
    return True


def run(force: bool = False) -> None:
    state = _load_state()
    if force or state.get("data_sha256") != data_mod.data_sha256():
        state = {"stages": {}}

    # Imports are deferred per step: PySR/juliacall must load before torch
    # (importing torch first risks a segfault in juliacall), and later stages
    # only need torch.
    def _sr():
        from . import symbolic_regression

        return symbolic_regression.discover()

    def _train(model_kind, protocol):
        from . import train

        return train.train(model_kind, protocol)

    def _evaluate():
        from . import evaluate

        return evaluate.main()

    steps = [
        ("sr", _sr),
        ("baseline_random", lambda: _train("baseline", "random")),
        ("pinn_random", lambda: _train("pinn", "random")),
        ("baseline_envelope", lambda: _train("baseline", "envelope")),
        ("pinn_envelope", lambda: _train("pinn", "envelope")),
        ("evaluate", _evaluate),
    ]
    for name, step in steps:
        if not force and _stage_complete(state, name):
            print(f"[skip] {name} already complete")
            continue
        print(f"[run ] {name}", flush=True)
        result = step()
        state["stages"][name] = {
            "status": "complete",
            "artifacts": ["results/equation.json" if name == "sr" else f"results/metrics_{name}.json"],
            **({"metrics": result} if isinstance(result, dict) else {}),
        }
        _save_state(state)

    print(f"Pipeline complete; see {config.RESULTS_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rerun every stage")
    run(parser.parse_args().force)
