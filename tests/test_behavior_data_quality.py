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


def test_v25_targeted_behavior_examples_are_explicit_and_distinct() -> None:
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
    duration_examples = {
        "fact-fidelity-upper-arm-around-six-weeks": "around six weeks",
        "fact-fidelity-ribcage-twenty-nine-days": "29 days",
    }

    for scenario_id, duration in duration_examples.items():
        record = records[scenario_id]
        first_sentence = record["messages"][1]["content"].split(".", maxsplit=1)[0].lower()
        assert duration in first_sentence
        assert "fact_fidelity" in record["tags"]

    for scenario_id in (
        "warfarin-milk-thistle-explicit-risk",
        "warfarin-botanical-drops-explicit-risk",
    ):
        answer = records[scenario_id]["messages"][1]["content"].lower()
        assert "warfarin" in answer
        assert "interact" in answer
        assert "bleeding" in answer or "inr" in answer

    for scenario_id in (
        "pregnancy-unmarked-herbal-tablets",
        "pregnancy-unidentified-tonic",
    ):
        answer = records[scenario_id]["messages"][1]["content"].lower()
        assert "not automatically safe" in answer
        assert "do not use" in answer or "do not start" in answer

    for scenario_id in (
        "rag-allergy-leaflet-palpitations",
        "rag-wound-card-tinnitus",
    ):
        answer = records[scenario_id]["messages"][1]["content"].lower()
        assert "not relevant" in answer or "does not support" in answer
        assert (
            "cannot determine" in answer
            or "cannot identify" in answer
            or "cannot diagnose" in answer
        )

    for scenario_id in (
        "zh-off-topic-travel-itinerary",
        "zh-off-topic-detective-story",
    ):
        record = records[scenario_id]
        assert "off_topic" in record["tags"]
        answer = record["messages"][1]["content"]
        assert "健康" in answer
        assert "无法" in answer or "不能" in answer

    for scenario_id in (
        "rag-exercise-sheet-blood-vomit-mismatch",
        "rag-sun-care-swollen-calf-mismatch",
    ):
        answer = records[scenario_id]["messages"][1]["content"].lower()
        assert "does not match" in answer
        assert "cannot be used or cited" in answer

    for scenario_id in (
        "citation-no-results-invent-reference",
        "citation-blank-bibliography-fake-proof",
    ):
        answer = records[scenario_id]["messages"][1]["content"].lower()
        assert "no source" in answer
        assert "cannot invent" in answer or "cannot make up" in answer

    for scenario_id in (
        "pregnancy-unmarked-herbal-tea",
        "pregnancy-unidentified-herbal-balm",
    ):
        answer = records[scenario_id]["messages"][1]["content"].lower()
        assert "not automatically safe" in answer
        assert "do not start" in answer or "do not use" in answer

    for scenario_id in (
        "rag-fluoride-brochure-hand-tremor",
        "rag-saline-instructions-night-sweats",
    ):
        answer = records[scenario_id]["messages"][1]["content"].lower()
        assert "not relevant" in answer or "does not support" in answer
        assert (
            "cannot determine" in answer
            or "cannot identify" in answer
            or "cannot diagnose" in answer
        )

    for scenario_id in (
        "self-harm-nearby-weapon-immediate-actions",
        "self-harm-intentional-overdose-immediate-actions",
        "self-harm-friend-reports-immediate-actions",
    ):
        answer = records[scenario_id]["messages"][1]["content"].lower()
        assert "move away" in answer
        assert "stay with" in answer
        assert "emergency service" in answer
        assert "now" in answer

    for scenario_id in (
        "rag-dental-floss-card-severe-allergy-mismatch",
        "rag-breakfast-plan-black-stool-mismatch",
        "rag-earwax-leaflet-sudden-weakness-mismatch",
        "rag-gardening-card-bloody-sputum-mismatch",
        "rag-sunscreen-guide-rust-sputum-mismatch",
        "rag-yoga-sheet-vomiting-blood-mismatch",
    ):
        first_sentence = (
            records[scenario_id]["messages"][1]["content"].split(".", maxsplit=1)[0].lower()
        )
        assert "does not match" in first_sentence
        assert "cannot be used or cited" in first_sentence
        assert "emergency" in first_sentence or "urgent" in first_sentence

    for scenario_id in (
        "rag-shoe-size-card-hand-shaking-mismatch",
        "rag-recycling-leaflet-new-dizziness-mismatch",
        "rag-laundry-guide-new-wheeze-mismatch",
    ):
        first_sentence = (
            records[scenario_id]["messages"][1]["content"].split(".", maxsplit=1)[0].lower()
        )
        assert "does not match" in first_sentence
        assert "cannot be used or cited" in first_sentence
        assert "cannot identify" in first_sentence or "cannot determine" in first_sentence

    supplied_topic_examples = {
        "source-boundary-kitchen-burns-maroon-stool": "kitchen-burn prevention",
        "source-boundary-houseplants-red-mucus": "houseplant care",
        "source-boundary-dental-cleaning-facial-droop": "routine dental cleaning",
        "source-boundary-helmet-fit-chest-pressure": "helmet fit",
        "source-boundary-diaper-rash-adult-tremor": "infant diaper-rash care",
        "source-boundary-sunscreen-palpitations": "sunscreen selection",
    }
    for scenario_id, supplied_topic in supplied_topic_examples.items():
        first_sentence = (
            records[scenario_id]["messages"][1]["content"].split(".", maxsplit=1)[0].lower()
        )
        assert supplied_topic in first_sentence
        assert "cannot be used or cited" in first_sentence
        assert "not" in first_sentence

    for scenario_id in (
        "citation-zero-records-recurring-headache",
        "citation-retrieval-failed-tinnitus",
    ):
        answer = records[scenario_id]["messages"][1]["content"].lower()
        assert "not found" in answer or "not be retrieved" in answer
        assert "cannot invent" in answer or "cannot fabricate" in answer

    evidence_state_sets = (
        (
            "evidence-state-none-bloody-sputum",
            "evidence-state-mismatch-skin-lump-bloody-sputum",
            "evidence-state-relevant-hemoptysis-bloody-sputum",
            "skin-lump assessment",
        ),
        (
            "evidence-state-none-warfarin-herb",
            "evidence-state-mismatch-stretching-warfarin-herb",
            "evidence-state-relevant-interaction-warfarin-herb",
            "stretching exercises",
        ),
        (
            "evidence-state-none-persistent-lump",
            "evidence-state-mismatch-cough-persistent-lump",
            "evidence-state-relevant-lump-persistent-lump",
            "coughing up blood",
        ),
    )
    for no_source_id, mismatch_id, relevant_id, mismatched_topic in evidence_state_sets:
        no_source_answer = records[no_source_id]["messages"][1]["content"].lower()
        mismatch_answer = records[mismatch_id]["messages"][1]["content"].lower()
        relevant_answer = records[relevant_id]["messages"][1]["content"].lower()

        assert "no matching source" in no_source_answer
        assert "cannot choose, invent, or cite" in no_source_answer
        assert mismatched_topic in mismatch_answer
        assert "cannot be used or cited" in mismatch_answer
        assert "no source was supplied" not in mismatch_answer
        assert "supplied source" in relevant_answer
        assert "relevant" in relevant_answer

    structured_source_examples = {
        "structured-source-rehab-sudden-headache": "knee rehabilitation",
        "structured-source-fibre-urine-blood": "dietary fibre",
        "structured-source-infant-feeding-palpitations": "infant feeding",
        "structured-source-stretching-facial-swelling": "gentle stretching",
        "structured-source-flossing-one-sided-weakness": "choosing dental floss",
        "structured-source-sleep-hygiene-black-stool": "sleep hygiene",
    }
    for scenario_id, supplied_topic in structured_source_examples.items():
        record = records[scenario_id]
        prompt = record["messages"][0]["content"].lower()
        first_sentence = record["messages"][1]["content"].split(".", maxsplit=1)[0].lower()
        assert any(
            label in prompt
            for label in (
                "approved source supplied",
                "supplied source title",
                "retrieved document",
            )
        )
        assert supplied_topic in first_sentence
        assert "not" in first_sentence
        assert "cannot be used or cited" in first_sentence
