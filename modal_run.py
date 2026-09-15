"""Run the complete leakage-safe chiller experiment on Modal and persist outputs."""

from __future__ import annotations

import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
APP_NAME = "pinn-chiller-clean"
VOLUME_NAME = "pinn-chiller-results"
REMOTE_REPO = "/root/pinn-dev"
REMOTE_OUTPUT = "/outputs/latest"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install(
        "numpy==2.5.2",
        "pandas==3.0.5",
        "scikit-learn==1.9.0",
        "scipy==1.18.1",
        "matplotlib==3.11.1",
        "sympy==1.14.0",
        "torch==2.13.0",
        "pysr==1.5.10",
        "gplearn==0.4.3",
    )
    .run_commands("python -c 'import pysr; print(\"pysr ready\")'")
    .add_local_dir(
        str(ROOT),
        remote_path=REMOTE_REPO,
        copy=True,
        ignore=[
            ".git/**",
            ".venv/**",
            "artifacts/**",
            "results/**",
            "__pycache__/**",
            "**/__pycache__/**",
            ".DS_Store",
        ],
    )
)


@app.function(
    image=image,
    gpu=["L4", "A10G", "T4"],
    cpu=8.0,
    memory=16384,
    timeout=6900,
    volumes={"/outputs": volume},
)
def run_experiment() -> str:
    import os
    import shutil
    import subprocess
    import sys
    import time

    import torch

    os.chdir(REMOTE_REPO)
    shutil.rmtree("results", ignore_errors=True)
    shutil.rmtree(REMOTE_OUTPUT, ignore_errors=True)
    Path(REMOTE_OUTPUT).mkdir(parents=True, exist_ok=True)

    print("torch:", torch.__version__, flush=True)
    print("cuda_available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), flush=True)
    print("cpu_count:", os.cpu_count(), flush=True)
    print("protocol: clean-v4-modal-bounded-pysr-20260915", flush=True)

    started = time.time()
    subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "scripts"], check=True)
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "src.run_all", "--force"], check=True)
    subprocess.run(
        [sys.executable, "scripts/publish_kaggle_artifacts.py", "--no-upload"],
        check=True,
    )

    shutil.copytree("results", f"{REMOTE_OUTPUT}/results", dirs_exist_ok=True)
    shutil.copy2("artifacts/kaggle/pinn_artifacts.tar.gz", f"{REMOTE_OUTPUT}/pinn_artifacts.tar.gz")
    summary = json.loads(Path("results/final_summary.json").read_text())
    run_metadata = {
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "summary": summary,
    }
    Path(f"{REMOTE_OUTPUT}/modal_run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    volume.commit()
    return json.dumps(run_metadata, indent=2)


@app.local_entrypoint()
def main():
    print(run_experiment.remote())
