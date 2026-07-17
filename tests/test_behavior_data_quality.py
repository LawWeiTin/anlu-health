import json
from pathlib import Path

from training.audit_behavior_dataset import audit


def test_behavior_dataset_passes_coverage_provenance_and_leakage_gates() -> None:
    report = audit(
        Path("training/data/sample_sft.jsonl"),
        Path("data/source_registry.yaml"),
        [
            Path("data/eval/golden_cases.jsonl"),
            Path("training/data/model_release_cases.jsonl"),
        ],
    )

    assert report["status"] == "pass", json.dumps(report["failures"], indent=2)
    assert report["records"] >= 45
    assert report["leakage"] == []
