from datetime import date, timedelta

from app.database import session_factory
from app.embeddings import MockMultilingualEmbeddings
from app.models import KnowledgeChunk, KnowledgeSource
from app.rag import Retriever


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
