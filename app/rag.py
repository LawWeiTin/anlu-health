import math
import re
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingProvider
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
_SOURCE_REFERENCE = re.compile(
    r"\b(?:article|card|document|evidence|retrieved|source|supplied)\b|资料|来源|检索|证据",
    re.I,
)
_FOCUS_MARKERS = (
    re.compile(r"\buser question\s*[:\-]\s*", re.I),
    re.compile(r"\bmy (?:actual )?question (?:is|about)\s*[:\-]?\s*", re.I),
    re.compile(r"\bbut\s+(?=(?:i|my)\b)", re.I),
    re.compile(r"但(?:我的问题|我)(?:是|关于)?"),
)

_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "hemoptysis",
        re.compile(
            r"\b(?:hemoptysis|cough(?:ing|ed|s)?(?: up)? blood|bloody|blood[- ]streaked)\b|"
            r"咳血|痰中带血",
            re.I,
        ),
    ),
    ("lumps_masses", re.compile(r"\b(?:lump|mass|nodule|bump|swelling)\b|肿块|包块|结节|肿胀", re.I)),
    (
        "medicine_interactions",
        re.compile(
            r"\b(?:interact(?:ion|ions)?|warfarin|anticoagulant|blood thinner|"
            r"mix(?:ing)? .{0,20}(?:medicine|medication|drug|herb)|"
            r"(?:medicine|medication|drug|herb).{0,20}(?:together|safe))\b|"
            r"药物相互作用|中西药同服|华法林|抗凝|同服",
            re.I,
        ),
    ),
    (
        "traditional_medicine",
        re.compile(
            r"\b(?:traditional chinese medicine|tcm|chinese herb|herbal formula|"
            r"acupuncture|tai chi|ginseng)\b|中医|中药|针灸|太极|人参",
            re.I,
        ),
    ),
    (
        "medicine_safety",
        re.compile(
            r"\b(?:medicine list|medication list|supplement list|side effect|allerg(?:y|ies)|"
            r"pregnan(?:t|cy)|surgery|liver disease|kidney disease)\b|"
            r"药物清单|用药安全|怀孕|手术|肝病|肾病",
            re.I,
        ),
    ),
    (
        "appointment_preparation",
        re.compile(r"\b(?:prepare|preparing|bring|notes?)\b.{0,30}\b(?:appointment|visit|doctor|clinician)\b|就诊准备", re.I),
    ),
    ("cough", re.compile(r"\b(?:cough|coughing|coughed)\b|咳嗽", re.I)),
)


def _detect_topics_in_text(text: str) -> frozenset[str]:
    matches = {topic for topic, pattern in _TOPIC_PATTERNS if pattern.search(text)}
    # Hemoptysis evidence is intentionally isolated from ordinary cough guidance.
    if "hemoptysis" in matches:
        matches.discard("cough")
    if "medicine_interactions" in matches:
        matches.discard("traditional_medicine")
        matches.discard("medicine_safety")
    return frozenset(matches)


def _clinical_focus_text(text: str) -> str:
    """Ignore a referenced source topic when the user clearly states a different question."""

    if not _SOURCE_REFERENCE.search(text):
        return text
    focus_matches = [
        match
        for pattern in _FOCUS_MARKERS
        for match in pattern.finditer(text)
    ]
    if focus_matches:
        marker = max(focus_matches, key=lambda item: item.end())
        focused = text[marker.end() :].strip()
        if focused:
            return focused
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？])\s*", text)
        if item.strip()
    ]
    return sentences[-1] if len(sentences) > 1 else text


def detect_topics(text: str) -> frozenset[str]:
    """Return deterministic, auditable topic labels for a user's clinical focus."""

    return _detect_topics_in_text(_clinical_focus_text(text))


def _source_topics(source: KnowledgeSource, chunk: KnowledgeChunk) -> frozenset[str]:
    declared = frozenset(str(topic).strip() for topic in (source.topics or []) if str(topic).strip())
    # Derivation keeps pre-migration/test data safe, while ingestion requires explicit labels.
    return declared or _detect_topics_in_text(f"{source.title}\n{chunk.content}")


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
        query_topics = detect_topics(query)
        ranked: list[RetrievedChunk] = []
        for chunk, source, semantic in candidates:
            source_topics = _source_topics(source, chunk)
            if query_topics and not (query_topics & source_topics):
                continue
            chunk_tokens = _tokens(chunk.content)
            overlap = query_tokens & chunk_tokens
            # A vector match alone cannot authorize medical evidence. Unknown-topic queries
            # require direct lexical support; known topics have already passed the hard gate.
            if not overlap and not (query_topics & source_topics):
                continue
            lexical = len(overlap) / max(1, len(query_tokens))
            topic_match = bool(query_topics & source_topics)
            score = (
                (0.70 * max(0.0, semantic))
                + (0.14 * lexical)
                + (0.16 if topic_match else 0.0)
                + _EVIDENCE_BOOST.get(source.evidence_tier, 0.0)
            )
            if score >= self.min_score:
                ranked.append(RetrievedChunk(chunk=chunk, source=source, score=score))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]
