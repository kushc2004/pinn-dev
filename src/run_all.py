"""Run the leakage-safe end-to-end experiment with resumable stages."""

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
    state["protocol_version"] = config.PROTOCOL_VERSION
    state["split_audits"] = {
        protocol: data_mod.split_audit(protocol) for protocol in ("random", "high_load")
    }
    try:
        state["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        state["git_commit"] = None
    config.STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def _state_compatible(state: dict) -> bool:
    return (
        state.get("data_sha256") == data_mod.data_sha256()
        and state.get("protocol_version") == config.PROTOCOL_VERSION
    )


def _stage_complete(state: dict, name: str) -> bool:
    stage = state.get("stages", {}).get(name)
    return bool(_state_compatible(state) and stage and stage.get("status") == "complete")


def run(force: bool = False) -> None:
    state = _load_state()
    if force or not _state_compatible(state):
        state = {"stages": {}}

    # PySR/Juliacall discovery happens before importing torch training paths.
    def _sr(protocol):
        from . import symbolic_regression
        return symbolic_regression.discover(protocol, force=force)

    def _tune(protocol):
        from . import train
        return train.tune_lambda(protocol, force=force)

    def _final(protocol):
        from . import train
        return train.train_final(protocol, force=force)

    def _evaluate():
        from . import evaluate
        return evaluate.main()

    steps = [
        ("sr_random", lambda: _sr("random")),
        ("sr_high_load", lambda: _sr("high_load")),
        ("tune_random", lambda: _tune("random")),
        ("tune_high_load", lambda: _tune("high_load")),
        ("final_random", lambda: _final("random")),
        ("final_high_load", lambda: _final("high_load")),
        ("evaluate", _evaluate),
    ]
    for name, step in steps:
        if not force and _stage_complete(state, name):
            print(f"[skip] {name} already complete")
            continue
        print(f"[run ] {name}", flush=True)
        result = step()
        compact = result
        if isinstance(result, dict) and name.startswith("final_"):
            compact = {k: v for k, v in result.items() if k != "validation_records"}
        state.setdefault("stages", {})[name] = {"status": "complete", "result": compact}
        _save_state(state)

    print(f"Pipeline complete; see {config.RESULTS_DIR / 'final_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rerun every stage")
    run(parser.parse_args().force)
