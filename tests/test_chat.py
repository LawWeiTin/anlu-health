from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.chat import (
    UNVERIFIED_MODEL_RESPONSE,
    anchor_explicit_single_source,
    guard_generated_answer,
    validate_citations,
)
from app.database import session_factory
from app.embeddings import get_embedding_provider
from app.llm import MockMedicalModel, clean_provider_output
from app.models import KnowledgeChunk, KnowledgeSource
from app.rag import RetrievedChunk
from tests.conftest import csrf_headers


def _seed_source() -> None:
    content = (
        "An unexplained lump should be examined in person. Examples of possible patterns include "
        "a soft mobile lipoma, a smooth skin cyst, a painful hot abscess with fever, or a swollen "
        "gland. These examples are not a diagnosis. Record growth, pain, warmth, and mobility."
    )
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
        "Possible causes include irritation after a severe cough, bronchitis, pneumonia, tuberculosis, "
        "bronchiectasis, a blood clot in the lung, or lung cancer; examples are not a diagnosis. "
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


def _seed_pregnancy_nutrition_source() -> None:
    content = (
        "Pregnancy nutrition can include vegetables, fruit, whole grains, protein foods, and dairy "
        "or fortified soy. Iron-rich examples include lean meat, seafood, poultry, fortified cereal "
        "and bread, white beans, lentils, spinach, kidney beans, peas, nuts, and raisins. Vitamin C "
        "foods such as strawberries, peppers, tomatoes, and broccoli can be paired with plant iron. "
        "Example combinations include fortified cereal with strawberries and lentils with tomatoes."
    )
    vector = get_embedding_provider().embed([content])[0]
    with session_factory()() as db:
        source = KnowledgeSource(
            source_key="test-pregnancy-nutrition",
            title="Pregnancy diet and iron-rich food examples",
            publisher="Test public health authority",
            url="https://example.gov/pregnancy-nutrition",
            license="Public domain",
            evidence_tier="government_consumer",
            topics=["pregnancy_nutrition", "iron_nutrition"],
            keywords=["pregnancy diet", "iron foods", "lentils", "fortified cereal"],
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
                token_count=88,
                embedding=vector,
            )
        )
        db.commit()


def test_emergency_chat_does_not_need_rag(registered_client: TestClient) -> None:
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={
            "message": "I cannot breathe and my lips are blue",
            "care_mode": "integrative",
            "medical_disclaimer_accepted": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "emergency"
    assert body["evidence_status"] == "safety_bypass"
    assert "model generation were bypassed" in body["evidence_notice"]
    assert body["sources"] == []
    assert "995" in body["answer"]


def test_non_emergency_abstains_without_sources(registered_client: TestClient) -> None:
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={
            "message": "How should I prepare for an appointment?",
            "care_mode": "biomedical",
            "medical_disclaimer_accepted": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "could not find an approved" in body["answer"]
    assert body["evidence_status"] == "insufficient_sources"


def test_lump_answer_has_source_and_escalation(registered_client: TestClient) -> None:
    _seed_source()
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={
            "message": "I found a lump under my arm",
            "care_mode": "integrative",
            "medical_disclaimer_accepted": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "soon"
    assert body["evidence_status"] == "grounded"
    assert body["sources"][0]["id"] == "S1"
    assert "cannot" in body["answer"].casefold()
    assert "lipoma" in body["answer"].casefold()
    assert "abscess" in body["answer"].casefold()
    assert body["history_saved"] is False


def test_unsafe_model_draft_reports_rejected_evidence_state(
    registered_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnsafeModel:
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            return "Take 10 mg each morning. [S1]"

    _seed_source()
    monkeypatch.setattr("app.api.get_model_provider", lambda: UnsafeModel())

    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={
            "message": "I found a lump under my arm",
            "care_mode": "biomedical",
            "medical_disclaimer_accepted": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == UNVERIFIED_MODEL_RESPONSE
    assert body["evidence_status"] == "model_rejected"
    assert body["sources"] == []


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
            "medical_disclaimer_accepted": True,
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
    assert "bronchitis" in body["answer"].casefold()
    assert "blood clot" in body["answer"].casefold()
    assert "traditional chinese medicine perspective" not in body["answer"].casefold()


def test_pregnancy_nutrition_answer_names_foods_and_practical_examples(
    registered_client: TestClient,
) -> None:
    _seed_pregnancy_nutrition_source()

    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={
            "message": "What diet should I have during pregnancy, including high iron foods?",
            "care_mode": "biomedical",
            "medical_disclaimer_accepted": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_status"] == "grounded"
    assert "lentils" in body["answer"].casefold()
    assert "fortified cereal with strawberries" in body["answer"].casefold()
    assert "not a personalized diet" in body["answer"].casefold()


def test_small_talk_does_not_trigger_medical_boilerplate(registered_client: TestClient) -> None:
    response = registered_client.post(
        "/api/chat",
        headers=csrf_headers(registered_client),
        json={
            "message": "Hello how are you",
            "care_mode": "integrative",
            "medical_disclaimer_accepted": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "routine"
    assert body["evidence_status"] == "not_applicable"
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
        json={
            "message": "Write me a poem about the ocean.",
            "care_mode": "integrative",
            "medical_disclaimer_accepted": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_status"] == "not_applicable"
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


def test_explicit_relevance_can_anchor_only_one_supplied_source() -> None:
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
    retrieved = [RetrievedChunk(chunk, source, 1)]

    anchored = anchor_explicit_single_source(
        "The supplied source is relevant. Arrange an in-person review soon.",
        retrieved,
    )

    assert anchored.endswith("[S1].")
    assert anchor_explicit_single_source("This answer is unsupported.", retrieved) == (
        "This answer is unsupported."
    )
    assert anchor_explicit_single_source(
        "The supplied source is relevant.",
        retrieved * 2,
    ) == "The supplied source is relevant."


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


def test_incomplete_or_meta_model_output_fails_closed() -> None:
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
    retrieved = [RetrievedChunk(chunk, source, 1)]

    incomplete = guard_generated_answer("Contact a clinician because [S1]", retrieved)
    leaked = guard_generated_answer(
        "Choose an approved response before answering. [S1]",
        retrieved,
    )

    assert incomplete.text == UNVERIFIED_MODEL_RESPONSE
    assert leaked.text == UNVERIFIED_MODEL_RESPONSE


def test_provider_output_keeps_only_first_medgemma_turn() -> None:
    raw = "Seek medical care today.<end_of_turn><start_of_turn>model\nIgnore this."

    assert clean_provider_output(raw) == "Seek medical care today."


def test_provider_output_removes_medgemma_reasoning_markers() -> None:
    raw = (
        "<unused94>thought\nInternal reasoning.<unused95>model\n"
        "Seek medical care today.<end_of_turn>"
    )

    assert clean_provider_output(raw) == "Seek medical care today."


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
