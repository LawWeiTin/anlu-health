# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
"""Publish the checksum-verified V32 adapter from Kaggle to a private Hub repository.

Run this only inside a private Kaggle notebook that has the committed V32 output
attached as an input and exactly one encrypted secret named ``HF_DEPLOY_TOKEN``.
The token must have write access only to the destination adapter repository.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

DESTINATION_REPO = "lawwt/anlu-health-medgemma-v32-adapter"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
EXPECTED_BASE_MODEL = "google/medgemma-1.5-4b-it"
EXPECTED_SHA256 = {
    "README.md": "42c4bbc819284bf9c530a356018b9393f86bba34348039913597db5e2c4653d3",
    "adapter_config.json": "03d7835e3519b7d5cdc964cca641a9571bdd6d54300d7179af74bf2672277ebb",
    "adapter_model.safetensors": (
        "fa75cd09a14b4147b18d69d750dd7811845b25eba386d9f6881e0b6a67aa2ab5"
    ),
    "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
    "chat_template.jinja": "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
    "preprocessor_config.json": (
        "5b2f684b616a25f3cd4a700e5e471a07fcaabc5ec07871d2231b7e376e8648ce"
    ),
    "processor_config.json": (
        "3ffd5f11778dc73e2b69b3c00535e4121e1badf7018136263cd17b5b34fbaa53"
    ),
    "special_tokens_map.json": (
        "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397"
    ),
    "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
    "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    "tokenizer_config.json": (
        "120c4a25aed6436edaac2e3966c9aa026bc13b5f7d3d9a81fc7afe2569b9e28b"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_adapter(directory: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for filename, expected in EXPECTED_SHA256.items():
        path = directory / filename
        if not path.is_file():
            raise RuntimeError(f"Missing required V32 adapter file: {filename}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"V32 checksum mismatch for {filename}: expected {expected}, observed {actual}"
            )
        observed[filename] = actual

    adapter_config = json.loads((directory / "adapter_config.json").read_text("utf-8"))
    if adapter_config.get("base_model_name_or_path") != EXPECTED_BASE_MODEL:
        raise RuntimeError("V32 adapter base-model identity does not match the release registry")
    return observed


def find_verified_adapter(root: Path = KAGGLE_INPUT_ROOT) -> tuple[Path, dict[str, str]]:
    candidates = sorted(path.parent for path in root.rglob("adapter_model.safetensors"))
    if not candidates:
        raise RuntimeError("Attach the private V32 Kaggle output as a notebook input")

    failures: list[str] = []
    for directory in candidates:
        try:
            return directory, verify_adapter(directory)
        except RuntimeError as exc:
            failures.append(str(exc))
    raise RuntimeError("No checksum-verified V32 adapter input was found: " + "; ".join(failures))


def verify_remote_adapter(
    *,
    repo_id: str,
    revision: str,
    token: str,
    download_file: Callable[..., str],
    temporary_root: Path = Path("/kaggle/working"),
) -> dict[str, str]:
    temporary_root.mkdir(parents=True, exist_ok=True)
    verify_dir = Path(
        tempfile.mkdtemp(prefix=".anlu-v32-hub-verify-", dir=str(temporary_root))
    )
    try:
        for filename in EXPECTED_SHA256:
            download_file(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                revision=revision,
                token=token,
                local_dir=verify_dir,
                force_download=True,
            )
        return verify_adapter(verify_dir)
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)


def main() -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_DEPLOY_TOKEN")
    if not token:
        raise RuntimeError("Enable the encrypted Kaggle secret HF_DEPLOY_TOKEN")

    adapter_dir, checksums = find_verified_adapter()
    api = HfApi(token=token)
    repo = api.repo_info(DESTINATION_REPO, repo_type="model")
    if not repo.private:
        raise RuntimeError("Destination adapter repository must remain private")

    existing_files = {item.rfilename for item in repo.siblings}
    if set(EXPECTED_SHA256).issubset(existing_files):
        commit_oid = repo.sha
        status = "already_published_and_verified"
    else:
        operations = [
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=adapter_dir / filename)
            for filename in EXPECTED_SHA256
        ]
        commit = api.create_commit(
            repo_id=DESTINATION_REPO,
            repo_type="model",
            operations=operations,
            commit_message="Publish checksum-verified Anlu Health MedGemma V32 adapter",
        )
        commit_oid = commit.oid
        status = "published_and_verified"

    # The revision is the immutable commit returned by the Hub or its current HEAD.
    remote_checksums = verify_remote_adapter(
        repo_id=DESTINATION_REPO,
        revision=commit_oid,
        token=token,
        download_file=hf_hub_download,  # nosec B615
    )

    print(
        json.dumps(
            {
                "status": status,
                "private_repository": DESTINATION_REPO,
                "commit_oid": commit_oid,
                "file_count": len(remote_checksums),
                "checksums": checksums,
                "secret_value_printed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
