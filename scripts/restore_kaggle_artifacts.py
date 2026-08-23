#!/usr/bin/env python3
"""Restore pipeline results/checkpoints from a packaged artifact archive.

Verifies the embedded manifest checksums before touching results/, so a
corrupt cache can never silently replace good local state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _find_archive(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    candidates = [
        *Path("/kaggle/input").glob("*/pinn_artifacts.tar.gz"),
        ROOT / "artifacts/kaggle/pinn_artifacts.tar.gz",
        ROOT / "pinn_artifacts.tar.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No pinn_artifacts.tar.gz found; pass --archive or attach the artifact dataset"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="Path to pinn_artifacts.tar.gz (auto-discovered by default)")
    parser.add_argument("--dest", type=Path, default=ROOT, help="Repo root to restore into")
    arguments = parser.parse_args()

    archive_path = _find_archive(arguments.archive)
    print(f"Restoring from {archive_path}")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        raw_manifest = members.get("artifact_manifest.json")
        if raw_manifest is None:
            print("Archive carries no embedded manifest; restoring without verification")
            archive.extractall(arguments.dest, filter="data")
            return

        manifest = json.loads(archive.extractfile(raw_manifest).read())
        payloads: dict[str, bytes] = {}
        for record in sorted(manifest["files"], key=lambda r: r["path"]):
            relative = record["path"]
            member = members.get(relative)
            if member is None:
                raise RuntimeError(f"Archive is missing recorded file: {relative}")
            data = archive.extractfile(member).read()
            if _sha256_bytes(data) != record["sha256"]:
                raise RuntimeError(f"Checksum mismatch for {relative}; refusing to restore")
            payloads[relative] = data

        for relative, data in payloads.items():
            target = arguments.dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        print(f"Verified and restored {len(payloads)} files into {arguments.dest}")


if __name__ == "__main__":
    main()
