"""Build a deterministic, self-contained Kaggle release-candidate launcher."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

SNAPSHOT_FILES = (
    "training/medgemma_kaggle_qlora.py",
    "training/medgemma_format.py",
    "training/open_datasets.yaml",
    "training/prepare_open_datasets.py",
    "training/release_eval.py",
    "training/data/sample_sft.jsonl",
    "training/data/model_release_cases.jsonl",
)
KERNEL_ID = "lawweitin/anlu-health-medgemma-qlora-release-candidate"
KERNEL_TITLE = "Anlu Health - MedGemma QLoRA Release Candidate"
CODE_FILE = "anlu_medgemma_release_candidate.ipynb"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_launcher(repository_root: Path, commit: str) -> tuple[str, str]:
    """Return the standalone launcher and its deterministic manifest digest."""
    manifest: dict[str, str] = {}
    encoded_files: dict[str, str] = {}

    for relative in SNAPSHOT_FILES:
        payload = (repository_root / relative).read_bytes()
        manifest[relative] = _sha256(payload)
        encoded_files[relative] = base64.b64encode(payload).decode("ascii")

    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest_digest = _sha256(manifest_json.encode("utf-8"))
    snapshot_root = f"/kaggle/temp/anlu-private-snapshot-{manifest_digest[:12]}"

    launcher = f"""\
import base64
import hashlib
import os
import runpy
from pathlib import Path

COMMIT = {commit!r}
SNAPSHOT_MANIFEST = {manifest!r}
SNAPSHOT_FILES = {encoded_files!r}
SNAPSHOT_ROOT = Path({snapshot_root!r})


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


require(set(SNAPSHOT_FILES) == set(SNAPSHOT_MANIFEST), "Snapshot file manifest mismatch")
require(not SNAPSHOT_ROOT.exists(), f"Refusing to overwrite existing snapshot: {{SNAPSHOT_ROOT}}")
SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=False)

for relative, encoded in SNAPSHOT_FILES.items():
    relative_path = Path(relative)
    require(
        not relative_path.is_absolute() and ".." not in relative_path.parts,
        f"Unsafe snapshot path: {{relative}}",
    )
    payload = base64.b64decode(encoded, validate=True)
    actual = hashlib.sha256(payload).hexdigest()
    require(actual == SNAPSHOT_MANIFEST[relative], f"Checksum mismatch for {{relative}}")
    destination = SNAPSHOT_ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)

os.environ["ANLU_SNAPSHOT_DIR"] = str(SNAPSHOT_ROOT)
os.environ["ANLU_REPOSITORY_COMMIT"] = COMMIT
print(f"Verified private Anlu snapshot {{COMMIT}} at {{SNAPSHOT_ROOT}}")
runpy.run_path(
    str(SNAPSHOT_ROOT / "training/medgemma_kaggle_qlora.py"),
    run_name="__main__",
)
"""
    return launcher, manifest_digest


def write_release_bundle(
    repository_root: Path,
    output_dir: Path,
    commit: str,
) -> dict[str, object]:
    """Write the launcher and Kaggle metadata to an ignored staging directory."""
    launcher, manifest_digest = build_launcher(repository_root, commit)
    output_dir.mkdir(parents=True, exist_ok=True)

    code_path = output_dir / CODE_FILE
    metadata_path = output_dir / "kernel-metadata.json"
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": launcher.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    code_path.write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    metadata: dict[str, object] = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "code_file": code_path,
        "metadata_file": metadata_path,
        "manifest_sha256": manifest_digest,
        "launcher_sha256": _sha256(launcher.encode("utf-8")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_release_bundle(
        args.repository_root.resolve(),
        args.output_dir.resolve(),
        args.commit,
    )
    print(
        json.dumps(
            {
                "code_file": str(result["code_file"]),
                "metadata_file": str(result["metadata_file"]),
                "manifest_sha256": result["manifest_sha256"],
                "launcher_sha256": result["launcher_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
