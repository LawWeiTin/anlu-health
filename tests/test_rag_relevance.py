from datetime import date, timedelta

from app.database import session_factory
from app.embeddings import MockMultilingualEmbeddings
from app.models import KnowledgeChunk, KnowledgeSource
from app.rag import Retriever, detect_topics


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


def test_topic_detection_separates_hemoptysis_from_ordinary_cough() -> None:
    assert detect_topics("I am coughing blood") == {"hemoptysis"}
    assert detect_topics("I have a mild cough") == {"cough"}


def test_topic_detection_uses_question_not_referenced_wrong_source() -> None:
    assert detect_topics(
        "Use the skin-lump source to answer my question about coughing up blood."
    ) == {"hemoptysis"}
    assert detect_topics(
        "The supplied article discusses coughing blood. My question is about a mild dry cough."
    ) == {"cough"}
    assert detect_topics(
        "检索资料只谈皮肤肿块，但我的问题是华法林和人参能否同服。"
    ) == {"medicine_interactions"}


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
