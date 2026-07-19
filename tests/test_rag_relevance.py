import inspect
from datetime import date, timedelta

from app.database import session_factory
from app.embeddings import MockMultilingualEmbeddings
from app.models import KnowledgeChunk, KnowledgeSource
from app.rag import Retriever, _postgres_candidates
from app.search_text import knowledge_search_text, semantic_document_text


def test_mock_retrieval_abstains_when_topic_has_no_lexical_overlap() -> None:
    embeddings = MockMultilingualEmbeddings()
    content = "An unexplained skin lump or swelling should be examined in person."
    with session_factory()() as db:
        source = KnowledgeSource(
            source_key="lump-only",
            title="Skin lumps",
            publisher="Test authority",
            url="https://example.gov/lumps",
            license="Public domain",
            evidence_tier="government_consumer",
            topics=["lumps_masses"],
            language="en",
            reviewed_on=date.today(),
            expires_on=date.today() + timedelta(days=90),
            checksum_sha256="2" * 64,
            approved=True,
        )
        db.add(source)
        db.flush()
        db.add(
            KnowledgeChunk(
                source_id=source.id,
                ordinal=0,
                content=content,
                token_count=14,
                embedding=embeddings.embed([content])[0],
            )
        )
        db.commit()

        results = Retriever(embeddings).search(db, "I am coughing blood these past few days")

    assert results == []


def test_postgresql_path_combines_pgvector_and_full_text_search() -> None:
    source = inspect.getsource(_postgres_candidates)

    assert "cosine_distance" in source
    assert "to_tsvector" in source
    assert "to_tsquery" in source
    assert 'vector.op("@@")' in source


def test_hard_topic_filter_rejects_high_scoring_wrong_source() -> None:
    class ConstantEmbeddings(MockMultilingualEmbeddings):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * 383 for _ in texts]

    embeddings = ConstantEmbeddings()
    with session_factory()() as db:
        wrong = KnowledgeSource(
            source_key="high-score-lump",
            title="Skin lumps",
            publisher="Test authority",
            url="https://example.gov/lumps",
            license="Public domain",
            evidence_tier="guideline",
            topics=["lumps_masses"],
            language="en",
            reviewed_on=date.today(),
            expires_on=date.today() + timedelta(days=90),
            checksum_sha256="3" * 64,
            approved=True,
        )
        right = KnowledgeSource(
            source_key="lower-score-hemoptysis",
            title="Coughing up blood",
            publisher="Test authority",
            url="https://example.gov/hemoptysis",
            license="Public domain",
            evidence_tier="government_consumer",
            topics=["hemoptysis"],
            language="en",
            reviewed_on=date.today(),
            expires_on=date.today() + timedelta(days=90),
            checksum_sha256="4" * 64,
            approved=True,
        )
        db.add_all([wrong, right])
        db.flush()
        for source, content in (
            (wrong, "A lump should be examined. Blood tests are sometimes discussed."),
            (right, "Coughing up blood needs prompt medical assessment."),
        ):
            db.add(
                KnowledgeChunk(
                    source_id=source.id,
                    ordinal=0,
                    content=content,
                    token_count=12,
                    embedding=embeddings.embed([content])[0],
                )
            )
        db.commit()
        results = Retriever(embeddings).search(db, "I am coughing blood", limit=5)

    assert [result.source.source_key for result in results] == ["lower-score-hemoptysis"]


def test_strong_semantic_match_can_retrieve_a_paraphrase_without_exact_keywords() -> None:
    class SemanticEmbeddings(MockMultilingualEmbeddings):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * 383 for _ in texts]

    embeddings = SemanticEmbeddings()
    content = "Seek prompt assessment when respiratory bleeding is reported."
    with session_factory()() as db:
        source = KnowledgeSource(
            source_key="semantic-hemoptysis",
            title="Respiratory bleeding",
            publisher="Test authority",
            url="https://example.gov/respiratory-bleeding",
            license="Public domain",
            evidence_tier="government_consumer",
            topics=["hemoptysis"],
            keywords=["hemoptysis"],
            language="en",
            reviewed_on=date.today(),
            expires_on=date.today() + timedelta(days=90),
            checksum_sha256="5" * 64,
            approved=True,
        )
        db.add(source)
        db.flush()
        db.add(
            KnowledgeChunk(
                source_id=source.id,
                ordinal=0,
                content=content,
                search_text=knowledge_search_text(
                    title=source.title,
                    publisher=source.publisher,
                    topics=source.topics,
                    keywords=source.keywords,
                    content=content,
                ),
                token_count=10,
                embedding=embeddings.embed(
                    [
                        semantic_document_text(
                            title=source.title,
                            publisher=source.publisher,
                            topics=source.topics,
                            keywords=source.keywords,
                            content=content,
                        )
                    ]
                )[0],
            )
        )
        db.commit()
        results = Retriever(embeddings).search(
            db,
            "There are red streaks in material from my lungs",
        )

    assert [result.source.source_key for result in results] == ["semantic-hemoptysis"]
