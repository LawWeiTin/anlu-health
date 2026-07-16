from pathlib import Path

from scripts.evaluate_rag import evaluate


def test_rag_release_cases_have_perfect_topic_isolation_and_abstention() -> None:
    metrics = evaluate(
        Path("data/seed_knowledge.jsonl"),
        Path("data/eval/rag_cases.jsonl"),
    )

    assert metrics["expected_source_recall"] == 1.0
    assert metrics["forbidden_source_rate"] == 0.0
    assert metrics["abstention_accuracy"] == 1.0
