"""Offline topic-isolation and abstention release gate for the approved RAG seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import Base  # noqa: E402
from app.embeddings import MockMultilingualEmbeddings  # noqa: E402
from app.models import KnowledgeChunk, KnowledgeSource  # noqa: E402
from app.rag import Retriever  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(seed_path: Path, cases_path: Path) -> dict[str, Any]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    embeddings = MockMultilingualEmbeddings()
    with Session(engine) as db:
        for document in _read_jsonl(seed_path):
            content = document["content"]
            source = KnowledgeSource(
                source_key=document["source_key"],
                title=document["title"],
                publisher=document["publisher"],
                url=document["url"],
                license=document["license"],
                evidence_tier=document["evidence_tier"],
                topics=document["topics"],
                language=document["language"],
                reviewed_on=date.fromisoformat(document["reviewed_on"]),
                expires_on=date.fromisoformat(document["expires_on"]),
                checksum_sha256=hashlib.sha256(content.encode()).hexdigest(),
                approved=document["approved"],
            )
            db.add(source)
            db.flush()
            db.add(
                KnowledgeChunk(
                    source_id=source.id,
                    ordinal=0,
                    content=content,
                    token_count=max(1, len(content) // 4),
                    embedding=embeddings.embed([content])[0],
                )
            )
        db.commit()

        rows = []
        expected_hits = 0
        expected_total = 0
        forbidden_hits = 0
        abstention_correct = 0
        cases = _read_jsonl(cases_path)
        retriever = Retriever(embeddings)
        for case in cases:
            results = retriever.search(db, case["query"], limit=5)
            actual = [item.source.source_key for item in results]
            expected = set(case["expected_source_keys"])
            forbidden = set(case["forbidden_source_keys"])
            expected_total += len(expected)
            expected_hits += len(expected & set(actual))
            forbidden_hits += len(forbidden & set(actual))
            abstention_correct += (not actual) == bool(case["expect_abstention"])
            rows.append(
                {
                    "id": case["id"],
                    "actual_source_keys": actual,
                    "expected_source_keys": sorted(expected),
                    "forbidden_source_keys_found": sorted(forbidden & set(actual)),
                }
            )

    total = len(cases)
    return {
        "cases": total,
        "expected_source_recall": expected_hits / max(1, expected_total),
        "forbidden_source_rate": forbidden_hits / max(1, total),
        "abstention_accuracy": abstention_correct / max(1, total),
        "details": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=ROOT / "data" / "seed_knowledge.jsonl")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "eval" / "rag_cases.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    metrics = evaluate(args.seed, args.cases)
    rendered = json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not args.strict:
        return 0
    thresholds = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["rag_evaluation"]
    failures = (
        metrics["expected_source_recall"] < thresholds["expected_source_recall_min"],
        metrics["forbidden_source_rate"] > thresholds["forbidden_source_rate_max"],
        metrics["abstention_accuracy"] < thresholds["abstention_accuracy_min"],
    )
    return 1 if any(failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
