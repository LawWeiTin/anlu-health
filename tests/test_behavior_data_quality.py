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
    assert report["records"] >= 61
    assert report["leakage"] == []


def test_v18_duration_fidelity_targets_repeat_the_supplied_duration_first() -> None:
    records = {
        row["scenario_id"]: row
        for row in (
            json.loads(line)
            for line in Path("training/data/sample_sft.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    expected = {
        "fact-fidelity-shoulder-twenty-seven-days": "27 days",
        "fact-fidelity-lower-leg-four-weeks": "four weeks",
        "fact-fidelity-jawline-sixteen-days": "sixteen days",
        "fact-fidelity-back-thirty-one-days": "31 days",
    }

    for scenario_id, duration in expected.items():
        record = records[scenario_id]
        first_sentence = record["messages"][1]["content"].split(".", maxsplit=1)[0].lower()
        assert duration in first_sentence
        assert "fact_fidelity" in record["tags"]
