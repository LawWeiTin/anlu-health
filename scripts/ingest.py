"""Validate, chunk, embed, and idempotently ingest approved JSONL knowledge sources."""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import delete, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import Base, get_engine, session_factory  # noqa: E402
from app.embeddings import get_embedding_provider  # noqa: E402
from app.models import KnowledgeChunk, KnowledgeSource, utcnow  # noqa: E402
from app.search_text import knowledge_search_text, semantic_document_text  # noqa: E402


def load_registry(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["sources"]


def registry_entry(entries: list[dict[str, Any]], source_key: str) -> dict[str, Any] | None:
    for entry in entries:
        if entry["source_key"] == source_key:
            return entry
        prefix = entry.get("source_key_prefix")
        if prefix and source_key.startswith(prefix):
            return entry
    return None


def chunk_text(text: str, size: int = 1200, overlap: int = 160) -> list[str]:
    normalized = " ".join(text.split())
    if len(normalized) <= size:
        return [normalized]
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            boundary = max(normalized.rfind(". ", start, end), normalized.rfind("。", start, end))
            if boundary > start + (size // 2):
                end = boundary + 1
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def validate_document(document: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "source_key",
        "title",
        "publisher",
        "url",
        "license",
        "evidence_tier",
        "topics",
        "language",
        "reviewed_on",
        "expires_on",
        "approved",
        "content",
    }
    missing = required - document.keys()
    if missing:
        raise ValueError(f"{document.get('source_key', 'unknown')}: missing {sorted(missing)}")
    entry = registry_entry(entries, document["source_key"])
    if not entry or not entry.get("approved"):
        raise ValueError(f"{document['source_key']}: source is not approved in the registry")
    if entry.get("use") == "evaluation_only":
        raise ValueError(f"{document['source_key']}: evaluation-only data cannot enter RAG")
    if document["publisher"] != entry["publisher"]:
        raise ValueError(f"{document['source_key']}: publisher does not match registry")
    if document["license"] != entry["license_label"]:
        raise ValueError(f"{document['source_key']}: license label does not match registry")
    if document["evidence_tier"] != entry["evidence_tier"]:
        raise ValueError(f"{document['source_key']}: evidence tier does not match registry")
    topics = document["topics"]
    if not isinstance(topics, list) or not topics or any(not isinstance(topic, str) for topic in topics):
        raise ValueError(f"{document['source_key']}: topics must be a non-empty string list")
    registered_topics = entry.get("topics")
    if not isinstance(registered_topics, list) or sorted(set(topics)) != sorted(set(registered_topics)):
        raise ValueError(f"{document['source_key']}: topics do not match registry")
    keywords = document.get("keywords", [])
    registered_keywords = entry.get("keywords", [])
    if (
        not isinstance(keywords, list)
        or any(not isinstance(keyword, str) or not keyword.strip() for keyword in keywords)
        or sorted(set(keywords)) != sorted(set(registered_keywords))
    ):
        raise ValueError(f"{document['source_key']}: keywords do not match registry")
    document["keywords"] = keywords
    reviewed = date.fromisoformat(document["reviewed_on"])
    expires = date.fromisoformat(document["expires_on"])
    if expires <= reviewed or expires < date.today():
        raise ValueError(
            f"{document['source_key']}: source is expired or has an invalid review window"
        )
    if len(document["content"].strip()) < 80:
        raise ValueError(f"{document['source_key']}: content is too short")
    document["reviewed_on"] = reviewed
    document["expires_on"] = expires
    document["approved"] = bool(document["approved"] and entry["approved"])
    return document


def ingest(path: Path, registry_path: Path, dry_run: bool = False) -> tuple[int, int]:
    entries = load_registry(registry_path)
    documents = [
        validate_document(json.loads(line), entries)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if dry_run:
        return len(documents), sum(len(chunk_text(item["content"])) for item in documents)

    Base.metadata.create_all(get_engine())
    embeddings = get_embedding_provider()
    source_count = 0
    chunk_count = 0
    with session_factory()() as db:
        for document in documents:
            checksum = hashlib.sha256(document["content"].encode("utf-8")).hexdigest()
            source = db.scalar(
                select(KnowledgeSource).where(KnowledgeSource.source_key == document["source_key"])
            )
            if source is None:
                source = KnowledgeSource(
                    source_key=document["source_key"],
                    title=document["title"],
                    publisher=document["publisher"],
                    url=document["url"],
                    license=document["license"],
                    evidence_tier=document["evidence_tier"],
                    topics=document["topics"],
                    keywords=document["keywords"],
                    language=document["language"],
                    reviewed_on=document["reviewed_on"],
                    expires_on=document["expires_on"],
                    checksum_sha256=checksum,
                    approved=document["approved"],
                )
                db.add(source)
            source.title = document["title"]
            source.publisher = document["publisher"]
            source.url = document["url"]
            source.license = document["license"]
            source.evidence_tier = document["evidence_tier"]
            source.topics = document["topics"]
            source.keywords = document["keywords"]
            source.language = document["language"]
            source.reviewed_on = document["reviewed_on"]
            source.expires_on = document["expires_on"]
            source.checksum_sha256 = checksum
            source.approved = document["approved"]
            source.updated_at = utcnow()
            db.flush()
            db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source_id == source.id))

            chunks = chunk_text(document["content"])
            search_documents = [
                knowledge_search_text(
                    title=document["title"],
                    publisher=document["publisher"],
                    topics=document["topics"],
                    keywords=document["keywords"],
                    content=content,
                )
                for content in chunks
            ]
            semantic_documents = [
                semantic_document_text(
                    title=document["title"],
                    publisher=document["publisher"],
                    topics=document["topics"],
                    keywords=document["keywords"],
                    content=content,
                )
                for content in chunks
            ]
            vectors = embeddings.embed(semantic_documents)
            for ordinal, (content, search_text, vector) in enumerate(
                zip(chunks, search_documents, vectors, strict=True)
            ):
                db.add(
                    KnowledgeChunk(
                        source_id=source.id,
                        ordinal=ordinal,
                        content=content,
                        search_text=search_text,
                        token_count=max(1, len(content) // 4),
                        embedding=vector,
                    )
                )
            source_count += 1
            chunk_count += len(chunks)
        db.commit()
    return source_count, chunk_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "source_registry.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sources, chunks = ingest(args.input, args.registry, args.dry_run)
    verb = "validated" if args.dry_run else "ingested"
    print(f"{verb} {sources} sources and {chunks} chunks")


if __name__ == "__main__":
    main()
