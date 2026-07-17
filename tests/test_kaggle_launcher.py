from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.build_kaggle_launcher import CODE_FILE, SNAPSHOT_FILES, write_release_bundle


def test_release_bundle_is_private_and_self_contained(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    commit = "a" * 40

    result = write_release_bundle(repository_root, tmp_path, commit)

    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((tmp_path / CODE_FILE).read_text(encoding="utf-8"))
    launcher = "".join(notebook["cells"][0]["source"])
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["kernel_type"] == "notebook"
    assert metadata["code_file"] == CODE_FILE
    assert commit in launcher
    assert all(relative in launcher for relative in SNAPSHOT_FILES)
    assert "HF_TOKEN" not in launcher
    assert result["launcher_sha256"] == hashlib.sha256(launcher.encode()).hexdigest()
