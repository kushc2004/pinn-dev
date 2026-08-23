#!/usr/bin/env python3
"""Stream a Kaggle notebook run's logs to stdout and a local append-only file."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

DEFAULT_SLUG = "kushchaudhari/pinn-chiller-pipeline"


def _run_kaggle(*args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["kaggle", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return result.returncode, result.stdout.rstrip()


def _status(slug: str) -> str:
    code, output = _run_kaggle("kernels", "status", slug)
    if code:
        return f"STATUS_COMMAND_FAILED: {output}"
    marker = 'status "'
    if marker in output:
        return output.split(marker, 1)[1].split('"', 1)[0]
    return output.splitlines()[-1] if output else "UNKNOWN"


def _write_line(output, line: str) -> None:
    print(line, flush=True)
    output.write(line + "\n")
    output.flush()


def _dump_logs(output) -> int:
    code, logs = _run_kaggle("kernels", "logs", DEFAULT_SLUG)
    timestamp = datetime.now(timezone.utc).isoformat()
    if code:
        _write_line(output, f"[{timestamp}] log command failed ({code}): {logs}")
        return code
    for line in logs.splitlines():
        _write_line(output, line)
    return 0


def watch(slug: str, output_path: Path, once: bool, interval: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        started = datetime.now(timezone.utc).isoformat()
        _write_line(output, f"===== Kaggle log watcher started {started} ({slug}) =====")
        if once:
            current_status = _status(slug)
            _write_line(output, f"[{datetime.now(timezone.utc).isoformat()}] status: {current_status}")
            code = _dump_logs(output)
            _write_line(output, f"[watch] stopped with status {current_status}")
            return 1 if "ERROR" in current_status.upper() else max(code, 0)

        _write_line(output, "[watch] following Kaggle live log stream; Ctrl-C stops this watcher only")
        process = subprocess.Popen(
            ["kaggle", "kernels", "logs", "--follow", slug],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            _write_line(output, line.rstrip())
        process.wait()
        while True:
            current_status = _status(slug)
            _write_line(output, f"[{datetime.now(timezone.utc).isoformat()}] status: {current_status}")
            if current_status in {"COMPLETE", "ERROR", "CANCELACKNOWLEDGED"} or "FAILED" in current_status.upper():
                break
            import time

            time.sleep(interval)
        _write_line(output, f"[watch] finished with status {current_status}")
        return 1 if "ERROR" in current_status.upper() else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--output", type=Path, default=Path("outputs/logs/kaggle_pinn.log"))
    parser.add_argument("--once", action="store_true", help="Poll once instead of following")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between status polls after the stream ends")
    arguments = parser.parse_args()
    raise SystemExit(watch(arguments.slug, arguments.output, arguments.once, arguments.interval))
