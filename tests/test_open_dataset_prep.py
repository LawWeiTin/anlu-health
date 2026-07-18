import json
from pathlib import Path

from training.prepare_open_datasets import (
    load_medquad,
    load_pubmedqa,
    split_by_group,
    truncate_to_complete_sentences,
)


def _spec(dataset_id: str) -> dict:
    base = {
        "id": dataset_id,
        "revision": "a" * 40,
        "license": "test-license",
        "homepage": "https://example.test/dataset",
    }
    if dataset_id == "medquad":
        base.update(
            {
                "allowed_directories": ["4_MPlus_Health_Topics_QA"],
                "allowed_question_types": ["information", "symptoms"],
            }
        )
    return base


def test_medquad_allows_education_and_rejects_treatment_and_dosing(tmp_path: Path) -> None:
    directory = tmp_path / "4_MPlus_Health_Topics_QA"
    directory.mkdir()
    directory.joinpath("sample.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<Document id="1" source="test" url="https://example.test/health-topic">
  <Focus>Example condition</Focus>
  <QAPairs>
    <QAPair pid="1"><Question qid="q1" qtype="information">What is this condition?</Question>
      <Answer>This is sufficiently long general educational information about a condition.</Answer></QAPair>
    <QAPair pid="2"><Question qid="q2" qtype="treatment">How is it treated?</Question>
      <Answer>This treatment answer is sufficiently long but must not enter the pilot.</Answer></QAPair>
    <QAPair pid="3"><Question qid="q3" qtype="symptoms">What medicine should I take?</Question>
      <Answer>Take a dose of 25 mg every day, which must be excluded from this pilot.</Answer></QAPair>
  </QAPairs>
</Document>""",
        encoding="utf-8",
    )

    records, counters = load_medquad(tmp_path, _spec("medquad"))

    assert [item["metadata"]["source_record_id"] for item in records] == ["q1"]
    assert counters["question_type_excluded"] == 1
    assert counters["dosing_excluded"] == 1


def test_pubmedqa_reserves_official_test_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    examples = {
        "train-pmid": {
            "QUESTION": "Does the intervention improve the measured outcome?",
            "CONTEXTS": ["A controlled study reported an improvement in the measured outcome."],
            "LONG_ANSWER": "The abstract supports an improvement, with study limitations",
            "final_decision": "yes",
        },
        "heldout-pmid": {
            "QUESTION": "Is the held-out result positive?",
            "CONTEXTS": ["This abstract is reserved for the official evaluation split."],
            "LONG_ANSWER": "This answer must never appear in the training bundle.",
            "final_decision": "no",
        },
    }
    data_dir.joinpath("ori_pqal.json").write_text(json.dumps(examples), encoding="utf-8")
    data_dir.joinpath("test_ground_truth.json").write_text(
        json.dumps({"heldout-pmid": "no"}), encoding="utf-8"
    )

    records, counters = load_pubmedqa(tmp_path, _spec("pubmedqa"))

    assert [item["metadata"]["pmid"] for item in records] == ["train-pmid"]
    assert records[0]["messages"][-1]["content"].endswith(".")
    assert counters["official_test_examples_excluded"] == 1


def test_group_split_prevents_focus_leakage() -> None:
    records = [
        {
            "messages": [{"role": "user", "content": f"question {index}"}],
            "metadata": {"group_id": group},
        }
        for index, group in enumerate(["condition-a", "condition-a", "condition-b", "condition-c"])
    ]

    train, validation = split_by_group(records, validation_fraction=0.5, seed=42)
    train_groups = {item["metadata"]["group_id"] for item in train}
    validation_groups = {item["metadata"]["group_id"] for item in validation}

    assert not train_groups & validation_groups


def test_long_training_targets_stop_at_a_complete_source_sentence() -> None:
    text = (
        "The first sentence provides concise evidence. "
        "The second sentence remains within the limit. "
        "The third sentence must be excluded because it exceeds the word budget."
    )

    truncated = truncate_to_complete_sentences(text, max_words=13)

    assert truncated == (
        "The first sentence provides concise evidence. "
        "The second sentence remains within the limit."
    )
    assert truncated.endswith(".")


def test_short_training_target_gets_missing_terminal_punctuation() -> None:
    text = "The abstract supports an association while important limitations remain"

    normalized = truncate_to_complete_sentences(text, max_words=75)

    assert normalized == f"{text}."


def test_existing_terminal_punctuation_is_preserved() -> None:
    text = "The evidence remains uncertain!"

    normalized = truncate_to_complete_sentences(text, max_words=75)

    assert normalized == text
