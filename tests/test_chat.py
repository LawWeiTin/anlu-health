from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.chat import UNVERIFIED_MODEL_RESPONSE, guard_generated_answer, validate_citations
from app.database import session_factory
from app.embeddings import get_embedding_provider
from app.llm import MockMedicalModel
from app.models import KnowledgeChunk, KnowledgeSource
from app.rag import RetrievedChunk
from tests.conftest import csrf_headers


def _seed_source() -> None:
    content = "An unexplained lump should be examined in person. Record growth and pain."
    vector = get_embedding_provider().embed([content])[0]
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
                content=content,
                token_count=18,
                embedding=vector,
            )
        )
        db.commit()


def _seed_hemoptysis_source() -> None:
    content = (
        "Coughing up blood or bloody mucus should be medically assessed even without other symptoms. "
        "Get medical help right away for more than a few teaspoons, bleeding that will not stop, "
        "chest pain, dizziness, or severe shortness of breath."
    )
    vector = get_embedding_provider().embed([content])[0]
    with session_factory()() as db:
        source = KnowledgeSource(
            source_key="test-hemoptysis",
            title="Coughing up blood — urgent care navigation summary",
            publisher="Test public health authority",
            url="https://example.gov/coughing-up-blood",
            license="Public domain",
            evidence_tier="government_consumer",
            language="en",
            reviewed_on=date.today(),
            expires_on=date.today() + timedelta(days=90),
            checksum_sha256="1" * 64,
            approved=True,
        )
        db.add(source)
        db.flush()
        db.add(
            KnowledgeChunk(
                source_id=source.id,
                ordinal=0,
                content=content,
                token_count=42,
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


def test_coughing_blood_is_urgent_and_uses_only_relevant_source(
    registered_client: TestClient,
) -> None:
    _seed_hemoptysis_source()
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={
            "message": "I am coughing blood these past few days. What should I do?",
            "care_mode": "integrative",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "urgent"
    assert body["safety_flags"] == ["hemoptysis"]
    assert len(body["sources"]) == 1
    assert "coughing up blood" in body["sources"][0]["title"].casefold()
    assert "lump" not in body["answer"].casefold()
    assert "same-day" in body["answer"].casefold()
    assert "traditional chinese medicine perspective" not in body["answer"].casefold()


def test_small_talk_does_not_trigger_medical_boilerplate(registered_client: TestClient) -> None:
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={"message": "Hello how are you", "care_mode": "integrative"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "routine"
    assert body["sources"] == []
    assert "hello" in body["answer"].casefold()
    assert "local experimental mode" in body["answer"].casefold()
    assert "what this may mean" not in body["answer"].casefold()


def test_off_topic_request_is_scoped_without_retrieval_or_model_boilerplate(
    registered_client: TestClient,
) -> None:
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={"message": "Write me a poem about the ocean.", "care_mode": "integrative"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert "focused on health" in body["answer"].casefold()
    assert "what this may mean" not in body["answer"].casefold()


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


def test_model_output_without_valid_citation_fails_closed() -> None:
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

    result = guard_generated_answer(
        "This is an unsupported medical answer.",
        [RetrievedChunk(chunk, source, 1)],
    )

    assert result.text == UNVERIFIED_MODEL_RESPONSE
    assert result.cited_chunks == []


def test_numeric_personalized_dose_fails_closed_even_with_valid_citation() -> None:
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

    result = guard_generated_answer(
        "Take 10 mg each morning. [S1]",
        [RetrievedChunk(chunk, source, 1)],
    )

    assert result.text == UNVERIFIED_MODEL_RESPONSE
    assert result.cited_chunks == []


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
