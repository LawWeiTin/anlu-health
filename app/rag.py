import math
import re
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingProvider, MockMultilingualEmbeddings
from app.models import KnowledgeChunk, KnowledgeSource


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    source: KnowledgeSource
    score: float


_EVIDENCE_BOOST = {
    "guideline": 0.08,
    "systematic_review": 0.07,
    "government_consumer": 0.06,
    "clinical_review": 0.04,
    "traditional_framework": 0.0,
}
_ENGLISH_STOPWORDS = {
    "a",
    "am",
    "an",
    "and",
    "are",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "or",
    "should",
    "the",
    "these",
    "this",
    "to",
    "what",
    "with",
    "you",
}
_CJK_SEQUENCE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def _tokens(text: str) -> set[str]:
    lowered = text.casefold()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", lowered)
        if len(token) >= 2 and token not in _ENGLISH_STOPWORDS
    }
    for sequence in _CJK_SEQUENCE.findall(lowered):
        tokens.add(sequence)
        tokens.update(sequence[index : index + 2] for index in range(max(0, len(sequence) - 1)))
    return tokens


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return dot / (left_norm * right_norm)


class Retriever:
    def __init__(self, embedding_provider: EmbeddingProvider, min_score: float = 0.18) -> None:
        self.embedding_provider = embedding_provider
        self.min_score = min_score

    def search(self, db: Session, query: str, limit: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embedding_provider.embed([query])[0]
        today = date.today()
        dialect = db.get_bind().dialect.name

        conditions = (
            KnowledgeSource.approved.is_(True),
            KnowledgeSource.expires_on >= today,
        )
        if dialect == "postgresql":
            distance = KnowledgeChunk.embedding.cosine_distance(query_embedding).label("distance")
            rows = db.execute(
                select(KnowledgeChunk, KnowledgeSource, distance)
                .join(KnowledgeSource)
                .where(*conditions)
                .order_by(distance)
                .limit(limit * 4)
            ).all()
            candidates = [
                (chunk, source, 1.0 - float(distance_value))
                for chunk, source, distance_value in rows
            ]
        else:
            rows = db.execute(
                select(KnowledgeChunk, KnowledgeSource)
                .join(KnowledgeSource)
                .where(*conditions)
                .limit(500)
            ).all()
            candidates = [
                (chunk, source, _cosine(query_embedding, list(chunk.embedding)))
                for chunk, source in rows
            ]

        query_tokens = _tokens(query)
        ranked: list[RetrievedChunk] = []
        for chunk, source, semantic in candidates:
            chunk_tokens = _tokens(chunk.content)
            overlap = query_tokens & chunk_tokens
            if isinstance(self.embedding_provider, MockMultilingualEmbeddings) and not overlap:
                continue
            lexical = len(overlap) / max(1, len(query_tokens))
            score = (
                (0.78 * semantic)
                + (0.22 * lexical)
                + _EVIDENCE_BOOST.get(source.evidence_tier, 0.0)
            )
            if score >= self.min_score:
                ranked.append(RetrievedChunk(chunk=chunk, source=source, score=score))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]
