"""Remove private service locations from release metadata before publication."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_URL = re.compile(r"https://www\.kaggle\.com/code/[^\s\"']+")
HF_ENDPOINT_URL = re.compile(r"https://[^\s\"']+\.endpoints\.huggingface\.cloud")


def _sanitize_json_value(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {item_key: _sanitize_json_value(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item, key) for item in value]
    if not isinstance(value, str):
        return value
    if key in {"parent_folder_id", "root_folder_id"}:
        return "redacted-private-folder-id"
    if key == "endpoint_url":
        return "private-endpoint-location-redacted"
    value = KAGGLE_URL.sub("private-kaggle-run://redacted", value)
    return HF_ENDPOINT_URL.sub("private-endpoint-location-redacted", value)


def _sanitize_json_file(path: Path) -> bool:
    original = json.loads(path.read_text(encoding="utf-8"))
    sanitized = _sanitize_json_value(original)
    if sanitized == original:
        return False
    path.write_text(json.dumps(sanitized, indent=2) + "\n", encoding="utf-8")
    return True


def _sanitize_yaml_file(path: Path) -> bool:
    original = yaml.safe_load(path.read_text(encoding="utf-8"))
    sanitized = _sanitize_json_value(original)
    if sanitized == original:
        return False
    path.write_text(yaml.safe_dump(sanitized, sort_keys=False), encoding="utf-8")
    return True


def main() -> None:
    changed: list[Path] = []
    for path in sorted((ROOT / "model_registry" / "runs").glob("*.json")):
        if _sanitize_json_file(path):
            changed.append(path)
    for path in (ROOT / "model_registry" / "production.yaml", ROOT / "storage" / "google_drive.yaml"):
        if _sanitize_yaml_file(path):
            changed.append(path)
    print(f"sanitized_private_metadata files={len(changed)}")


if __name__ == "__main__":
    main()
