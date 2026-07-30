import json
from pathlib import Path

import pytest

from training.publish_v32_adapter_to_hub import (
    EXPECTED_BASE_MODEL,
    EXPECTED_SHA256,
    find_verified_adapter,
    sha256,
    verify_adapter,
    verify_remote_adapter,
)


def _write_adapter(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED_SHA256:
        content = (
            json.dumps({"base_model_name_or_path": EXPECTED_BASE_MODEL}).encode()
            if filename == "adapter_config.json"
            else filename.encode()
        )
        (directory / filename).write_bytes(content)


def test_sha256_reads_file_in_binary_mode(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"anlu-v32")
    assert sha256(artifact) == "caf9b73c15f809826e0327d6825c2a8a486a517930a1535bbc7bb01f4ac7697a"


def test_verify_adapter_rejects_checksum_mismatch(tmp_path: Path) -> None:
    _write_adapter(tmp_path)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_adapter(tmp_path)


def test_find_verified_adapter_requires_attached_output(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Attach the private V32 Kaggle output"):
        find_verified_adapter(tmp_path)


def test_verify_remote_adapter_creates_and_cleans_temporary_root(
    tmp_path: Path, monkeypatch
) -> None:
    expected = {
        filename: (
            json.dumps({"base_model_name_or_path": EXPECTED_BASE_MODEL}).encode()
            if filename == "adapter_config.json"
            else filename.encode()
        )
        for filename in EXPECTED_SHA256
    }
    monkeypatch.setattr(
        "training.publish_v32_adapter_to_hub.EXPECTED_SHA256",
        {filename: __import__("hashlib").sha256(content).hexdigest() for filename, content in expected.items()},
    )
    temporary_root = tmp_path / "missing-root"

    def fake_download(*, filename: str, local_dir: Path, **_: object) -> str:
        target = Path(local_dir) / filename
        target.write_bytes(expected[filename])
        return str(target)

    observed = verify_remote_adapter(
        repo_id="lawwt/private-adapter",
        revision="immutable-commit",
        token="scoped-test-token",
        download_file=fake_download,
        temporary_root=temporary_root,
    )

    assert len(observed) == len(EXPECTED_SHA256)
    assert temporary_root.is_dir()
    assert list(temporary_root.iterdir()) == []
