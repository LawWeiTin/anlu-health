import json
from pathlib import Path

import yaml


def test_open_dataset_manifest_is_pinned_and_excludes_patient_chats() -> None:
    manifest = yaml.safe_load(Path("training/open_datasets.yaml").read_text(encoding="utf-8"))
    assert manifest["storage_policy"] == "remote_colab_and_private_drive_only"
    assert manifest["contains_user_conversations"] is False
    assert manifest["behavior_sampling_weight"] == 16
    datasets = {item["id"]: item for item in manifest["datasets"]}
    assert datasets["medquad"]["role"] == "train"
    assert datasets["pubmedqa"]["role"] == "train"
    assert datasets["medmcqa"]["role"] == "evaluation_only"
    assert all(len(item["revision"]) == 40 for item in datasets.values())
    assert any("Scraped online patient" in item for item in manifest["excluded_dataset_classes"])


def test_colab_pilot_is_additive_remote_only_and_never_promotes() -> None:
    notebook = json.loads(
        Path("training/medgemma_open_data_pilot_colab.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "/content/anlu-open-data-" in source
    assert "/content/drive/MyDrive/Anlu Health" in source
    assert "exist_ok=False" in source
    assert "Qwen/Qwen2.5-3B-Instruct" in source
    assert "STORAGE_MODE == 'mounted_drive'" in source
    assert "Refusing to replace existing Google Drive folder" in source
    assert "hemoptysis_urgent" in source and "casual_greeting" in source
    assert "warfarin_ginseng" in source
    assert "'promotion_allowed': False" in source
    assert "num_train_epochs=1" in source
    assert "rm -rf" not in source
    assert "shutil.rmtree" not in source
    assert "HF_TOKEN" in source and "GH_TOKEN" in source
