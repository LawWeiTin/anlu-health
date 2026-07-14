from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.chat import validate_citations
from app.database import session_factory
from app.embeddings import get_embedding_provider
from app.llm import MockMedicalModel
from app.models import KnowledgeChunk, KnowledgeSource
from app.rag import RetrievedChunk
from tests.conftest import csrf_headers


def _seed_source() -> None:
    vector = get_embedding_provider().embed(["lump swelling examination"])[0]
    with session_factory()() as db:
        source = KnowledgeSource(
            source_key="test-lumps",
            title="Reviewed lump guidance",
            publisher="Test public health authority",
            url="https://example.gov/lumps",
            license="Public domain",
            evidence_tier="government_consumer",
            language="en",
            reviewed_on=date.today(),
            expires_on=date.today() + timedelta(days=90),
            checksum_sha256="0" * 64,
            approved=True,
        )
        db.add(source)
        db.flush()
        db.add(
            KnowledgeChunk(
                source_id=source.id,
                ordinal=0,
                content="Unexplained lumps should be examined in person. Record growth and pain.",
                token_count=18,
                embedding=vector,
            )
        )
        db.commit()


def test_emergency_chat_does_not_need_rag(registered_client: TestClient) -> None:
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={"message": "I cannot breathe and my lips are blue", "care_mode": "integrative"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "emergency"
    assert body["sources"] == []
    assert "995" in body["answer"]


def test_non_emergency_abstains_without_sources(registered_client: TestClient) -> None:
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={"message": "How should I prepare for an appointment?", "care_mode": "biomedical"},
    )
    assert response.status_code == 200
    assert "could not find an approved" in response.json()["answer"]


def test_lump_answer_has_source_and_escalation(registered_client: TestClient) -> None:
    _seed_source()
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={"message": "I found a lump under my arm", "care_mode": "integrative"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "soon"
    assert body["sources"][0]["id"] == "S1"
    assert "cannot" in body["answer"].casefold()
    assert body["history_saved"] is False


def test_unknown_citation_is_removed() -> None:
    source = KnowledgeSource(
        id="source-id",
        source_key="source-key",
        title="Source",
        publisher="Publisher",
        url="https://example.gov",
        license="Public domain",
        evidence_tier="guideline",
        language="en",
        reviewed_on=date.today(),
        expires_on=date.today() + timedelta(days=90),
        checksum_sha256="0" * 64,
        approved=True,
    )
    chunk = KnowledgeChunk(
        id="chunk-id", source_id="source-id", ordinal=0, content="text", token_count=1, embedding=[]
    )
    result = validate_citations(
        "Supported [S1], invented [S99].", [RetrievedChunk(chunk, source, 1)]
    )
    assert "[S1]" in result.text
    assert "[S99]" not in result.text
    assert len(result.cited_chunks) == 1
    assert result.cited_chunks[0][0] == "S1"


def test_mock_model_uses_topic_relevant_tcm_citation() -> None:
    prompt = """<care_mode>integrative</care_mode>
<minimum_urgency>soon</minimum_urgency>
<question>I found a lump</question>
<approved_sources>
[S1] Skin lumps — care navigation summary
Publisher: NLM
Excerpt: lump examination

[S2] Traditional Chinese medicine — evidence and safety summary
Publisher: NCCIH
Excerpt: evidence varies and herbal products have safety risks
</approved_sources>"""
    answer = MockMedicalModel().generate("system", prompt)
    tcm_section = answer.split("Traditional Chinese medicine perspective", 1)[1]
    assert "[S2]" in tcm_section
    assert "[S1]" not in tcm_section
