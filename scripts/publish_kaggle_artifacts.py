#!/usr/bin/env python3
"""Package pipeline results/checkpoints and optionally publish them with the Kaggle CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "kushchaudhari/pinn-chiller-artifacts"
INCLUDE = ("results",)
EXPECTED_STAGES = (
    "sr_random",
    "sr_high_load",
    "tune_random",
    "tune_high_load",
    "final_random",
    "final_high_load",
    "evaluate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_complete_pipeline() -> dict:
    """Refuse to cache a failed or partial run: resume would then trust bad state."""
    state_path = ROOT / "results/state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing pipeline state: {state_path}")
    state = json.loads(state_path.read_text())
    if state.get("data_sha256"):
        from src.data import data_sha256

        if state["data_sha256"] != data_sha256():
            raise RuntimeError("Pipeline state predates the current dataset")
    incomplete = {
        name: state.get("stages", {}).get(name, {}).get("status", "missing")
        for name in EXPECTED_STAGES
    }
    broken = {name: status for name, status in incomplete.items() if status != "complete"}
    if broken:
        raise RuntimeError(f"Refusing to publish an incomplete pipeline cache: {broken}")
    return state


def _collect_files() -> list[Path]:
    files: list[Path] = []
    for relative in INCLUDE:
        directory = ROOT / relative
        if directory.is_dir():
            files.extend(
                sorted(
                    path
                    for path in directory.rglob("*")
                    if path.is_file()
                    and path.name != ".DS_Store"
                    and "sr-tmp" not in path.parts  # PySR scratch dir
                )
            )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--create", action="store_true", help="Create the dataset; otherwise publish a new version.")
    parser.add_argument("--public", action="store_true", help="Only applies with --create. Default is private.")
    parser.add_argument("--message", default="Refresh PINN chiller experiment artifacts")
    parser.add_argument("--no-upload", action="store_true", help="Package only.")
    parser.add_argument("--skip-completeness", action="store_true", help="Allow packaging a partial run (e.g. debugging).")
    arguments = parser.parse_args()

    if not arguments.skip_completeness:
        _require_complete_pipeline()

    files = _collect_files()
    if not files:
        raise RuntimeError("Nothing to package; run the pipeline first")

    manifest = {"files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files]}
    staging = ROOT / "artifacts/.kaggle-staging"
    published = ROOT / "artifacts/kaggle"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(published, ignore_errors=True)
    staging.mkdir(parents=True)

    archive_path = staging / "pinn_artifacts.tar.gz"
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT))
        # Ship the manifest inside the archive so restores can self-verify.
        import io

        manifest_bytes = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("artifact_manifest.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
    (staging / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": "PINN Chiller Artifacts", "id": arguments.dataset, "licenses": [{"name": "other"}]},
            indent=2,
        )
        + "\n"
    )

    # Verify the round-trip before publishing.
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {m.name for m in archive.getmembers() if m.isfile()}
        missing = sorted({record["path"] for record in manifest["files"]} - members)
        if missing:
            raise RuntimeError(f"Archive is missing expected files: {missing[:3]}")

    os.replace(staging, published)
    published_archive = published / archive_path.name
    print(f"packaged {published_archive} ({published_archive.stat().st_size} bytes, {len(files)} files)")
    if arguments.no_upload:
        return
    if arguments.create:
        command = ["kaggle", "datasets", "create", "-p", str(published)]
        if arguments.public:
            command.append("-u")
    else:
        command = ["kaggle", "datasets", "version", "-p", str(published), "-m", arguments.message]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
