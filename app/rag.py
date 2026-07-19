"""Hybrid semantic-vector and keyword retrieval over approved medical evidence."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingProvider
from app.models import KnowledgeChunk, KnowledgeSource
from app.search_text import (
    knowledge_search_text,
    retrieval_query_text,
    search_tokens,
    semantic_query_text,
)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    source: KnowledgeSource
    score: float


@dataclass(frozen=True)
class _Candidate:
    chunk: KnowledgeChunk
    source: KnowledgeSource
    semantic: float
    keyword: float


_EVIDENCE_BOOST = {
    "guideline": 0.05,
    "systematic_review": 0.045,
    "government_consumer": 0.04,
    "clinical_review": 0.03,
    "traditional_framework": 0.0,
}


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return dot / (left_norm * right_norm)


def _chunk_search_text(chunk: KnowledgeChunk, source: KnowledgeSource) -> str:
    if chunk.search_text:
        return chunk.search_text
    return knowledge_search_text(
        title=source.title,
        publisher=source.publisher,
        topics=source.topics or [],
        keywords=source.keywords or [],
        content=chunk.content,
    )


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    """Compute BM25 scores for the bounded SQLite/local evaluation corpus."""

    query_terms = search_tokens(query)
    if not query_terms or not documents:
        return [0.0] * len(documents)
    tokenized = [search_tokens(document) for document in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / max(1, len(lengths))
    document_frequency = Counter(
        term for tokens in tokenized for term in set(tokens)
    )
    query_frequency = Counter(query_terms)
    scores: list[float] = []
    for tokens, length in zip(tokenized, lengths, strict=True):
        frequencies = Counter(tokens)
        score = 0.0
        for term, query_count in query_frequency.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (
                    (len(documents) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
            )
            denominator = frequency + 1.2 * (
                1.0 - 0.75 + 0.75 * (length / max(1.0, average_length))
            )
            score += (
                inverse_document_frequency
                * ((frequency * 2.2) / denominator)
                * min(2, query_count)
            )
        scores.append(score)
    return scores


def _keyword_coverage(query: str, document: str) -> float:
    query_terms = set(search_tokens(query, include_bigrams=False))
    if not query_terms:
        return 0.0
    document_terms = set(search_tokens(document, include_bigrams=False))
    return len(query_terms & document_terms) / len(query_terms)


def _postgres_candidates(
    db: Session,
    *,
    query: str,
    query_embedding: list[float],
    today: date,
    candidate_limit: int,
) -> list[_Candidate]:
    conditions = (
        KnowledgeSource.approved.is_(True),
        KnowledgeSource.expires_on >= today,
    )
    distance = KnowledgeChunk.embedding.cosine_distance(query_embedding).label("distance")
    vector_rows = db.execute(
        select(KnowledgeChunk, KnowledgeSource, distance)
        .join(KnowledgeSource)
        .where(*conditions)
        .order_by(distance)
        .limit(candidate_limit)
    ).all()

    safe_terms = list(dict.fromkeys(search_tokens(query)))
    keyword_rows = []
    if safe_terms:
        tsquery_text = " | ".join(safe_terms)
        vector = func.to_tsvector("simple", KnowledgeChunk.search_text)
        tsquery = func.to_tsquery("simple", tsquery_text)
        keyword_rank = func.ts_rank_cd(vector, tsquery).label("keyword_rank")
        keyword_rows = db.execute(
            select(KnowledgeChunk, KnowledgeSource, distance, keyword_rank)
            .join(KnowledgeSource)
            .where(*conditions, vector.op("@@")(tsquery))
            .order_by(keyword_rank.desc())
            .limit(candidate_limit)
        ).all()

    by_id: dict[str, _Candidate] = {}
    for chunk, source, distance_value in vector_rows:
        by_id[chunk.id] = _Candidate(
            chunk=chunk,
            source=source,
            semantic=1.0 - float(distance_value),
            keyword=0.0,
        )
    for chunk, source, distance_value, keyword_rank_value in keyword_rows:
        by_id[chunk.id] = _Candidate(
            chunk=chunk,
            source=source,
            semantic=1.0 - float(distance_value),
            keyword=float(keyword_rank_value),
        )
    return list(by_id.values())


def _local_candidates(
    db: Session,
    *,
    query: str,
    query_embedding: list[float],
    today: date,
) -> list[_Candidate]:
    rows = db.execute(
        select(KnowledgeChunk, KnowledgeSource)
        .join(KnowledgeSource)
        .where(
            KnowledgeSource.approved.is_(True),
            KnowledgeSource.expires_on >= today,
        )
        .limit(500)
    ).all()
    documents = [_chunk_search_text(chunk, source) for chunk, source in rows]
    keyword_scores = _bm25_scores(query, documents)
    return [
        _Candidate(
            chunk=chunk,
            source=source,
            semantic=_cosine(query_embedding, list(chunk.embedding)),
            keyword=keyword,
        )
        for (chunk, source), keyword in zip(rows, keyword_scores, strict=True)
    ]


class Retriever:
    """Fuse semantic vector similarity with BM25/PostgreSQL full-text ranking."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        min_score: float = 0.28,
        strong_semantic_score: float = 0.58,
        min_keyword_coverage: float = 0.25,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.min_score = min_score
        self.strong_semantic_score = strong_semantic_score
        self.min_keyword_coverage = min_keyword_coverage

    def search(self, db: Session, query: str, limit: int = 5) -> list[RetrievedChunk]:
        retrieval_query = retrieval_query_text(query)
        if not search_tokens(retrieval_query):
            return []
        query_embedding = self.embedding_provider.embed(
            [semantic_query_text(retrieval_query)]
        )[0]
        today = date.today()
        dialect = db.get_bind().dialect.name
        if dialect == "postgresql":
            candidates = _postgres_candidates(
                db,
                query=retrieval_query,
                query_embedding=query_embedding,
                today=today,
                candidate_limit=max(20, limit * 8),
            )
        else:
            candidates = _local_candidates(
                db,
                query=retrieval_query,
                query_embedding=query_embedding,
                today=today,
            )
        if not candidates:
            return []

        semantic_ranking = {
            candidate.chunk.id: rank
            for rank, candidate in enumerate(
                sorted(candidates, key=lambda item: item.semantic, reverse=True),
                start=1,
            )
        }
        keyword_candidates = [candidate for candidate in candidates if candidate.keyword > 0]
        keyword_ranking = {
            candidate.chunk.id: rank
            for rank, candidate in enumerate(
                sorted(keyword_candidates, key=lambda item: item.keyword, reverse=True),
                start=1,
            )
        }
        max_keyword = max((item.keyword for item in candidates), default=0.0)
        scored: list[tuple[RetrievedChunk, float]] = []
        for candidate in candidates:
            document = _chunk_search_text(candidate.chunk, candidate.source)
            coverage = _keyword_coverage(retrieval_query, document)
            semantic = max(0.0, candidate.semantic)
            keyword = candidate.keyword / max_keyword if max_keyword else 0.0
            vector_rrf = 11.0 / (10.0 + semantic_ranking[candidate.chunk.id])
            keyword_rank = keyword_ranking.get(candidate.chunk.id)
            keyword_rrf = 11.0 / (10.0 + keyword_rank) if keyword_rank else 0.0
            rank_fusion = (0.65 * vector_rrf) + (0.35 * keyword_rrf)
            score = (
                (0.48 * semantic)
                + (0.32 * keyword)
                + (0.15 * rank_fusion)
                + _EVIDENCE_BOOST.get(candidate.source.evidence_tier, 0.0)
            )
            has_lexical_support = (
                candidate.keyword > 0 and coverage >= self.min_keyword_coverage
            )
            if (
                score >= self.min_score
                and (has_lexical_support or semantic >= self.strong_semantic_score)
            ):
                scored.append(
                    (
                        RetrievedChunk(
                            chunk=candidate.chunk,
                            source=candidate.source,
                            score=score,
                        ),
                        coverage,
                    )
                )

        scored.sort(key=lambda item: (item[0].score, item[1]), reverse=True)
        if not scored:
            return []
        best_score = scored[0][0].score
        # Suppress weak tail matches that happen to share generic medical vocabulary.
        return [
            item
            for item, _coverage in scored
            if item.score >= best_score - 0.12
        ][:limit]
