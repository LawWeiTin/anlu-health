import logging
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.llm import ModelProvider
from app.models import ChatMessage, Conversation, User, utcnow
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.rag import RetrievedChunk, Retriever
from app.safety import SafetyAssessment, assess, emergency_response
from app.schemas import ChatResponse, SourceOut
from app.security import Cipher, short_fingerprint

logger = logging.getLogger("anlu.audit")
DISCLAIMER = (
    "Educational information only—not a diagnosis or treatment plan. For emergencies, contact local "
    "emergency services; for personal medical decisions, consult a qualified clinician."
)
UNVERIFIED_MODEL_RESPONSE = (
    "The model response did not pass the evidence and safety checks, so I will not present it as "
    "reliable. Please ask a qualified clinician or pharmacist. If symptoms are worsening or you "
    "are worried, seek in-person care."
)
_SMALL_TALK = re.compile(
    r"^\s*(?:hello|hi|hey|good (?:morning|afternoon|evening))"
    r"(?:[!,. ]+(?:how are you|how is it going))?[!?. ]*$|"
    r"^\s*(?:thanks|thank you|bye|goodbye)[!?. ]*$",
    re.I,
)
_OFF_TOPIC_REQUEST = re.compile(
    r"\b(?:write|compose|create)\b.{0,30}\b(?:poem|story|song|recipe)\b|"
    r"\b(?:debug|repair|write)\b.{0,30}\b(?:code|javascript|python|program)\b|"
    r"\b(?:stock|investment|weather|travel itinerary)\b|"
    r"写.{0,12}(?:诗|故事)|菜谱|股票|天气",
    re.I,
)
_HEALTH_CONTEXT = re.compile(
    r"\b(?:allerg|blood|breath|clinician|cough|diagnos|diet|doctor|drug|fever|health|"
    r"herb|lump|medical|medicine|nutrition|pain|pharmacist|pregnan|rash|symptom|tcm)\w*\b|"
    r"健康|医疗|医生|药|中医|症状|疼痛|咳|肿块|皮疹|怀孕",
    re.I,
)
_UNSAFE_MODEL_PATTERNS = (
    re.compile(
        r"\b(?:take|use|start)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|tablets?|capsules?)\b",
        re.I,
    ),
    re.compile(r"\byou (?:definitely|certainly) have\b", re.I),
    re.compile(r"\b(?:the|your) diagnosis is\b", re.I),
    re.compile(r"\bchoose a non[- ]controversial\b", re.I),
    re.compile(r"\bthe ai has been instructed\b", re.I),
    re.compile(r"\bi should up[- ]rise\b", re.I),
)


@dataclass(frozen=True)
class ValidatedAnswer:
    text: str
    cited_chunks: list[tuple[str, RetrievedChunk]]


def validate_citations(answer: str, chunks: list[RetrievedChunk]) -> ValidatedAnswer:
    allowed = {f"S{index}": chunk for index, chunk in enumerate(chunks, start=1)}
    cited: list[tuple[str, RetrievedChunk]] = []

    def replace(match: re.Match[str]) -> str:
        citation_id = match.group(1).upper()
        chunk = allowed.get(citation_id)
        if not chunk:
            return ""
        if all(existing.source.id != chunk.source.id for _, existing in cited):
            cited.append((citation_id, chunk))
        return f"[{citation_id}]"

    cleaned = re.sub(r"\[(S\d+)\]", replace, answer, flags=re.IGNORECASE)
    if chunks and not cited:
        cleaned += (
            "\n\nEvidence note\nThe generated text did not include a verifiable source citation, so "
            "treat it as unverified and ask a qualified clinician."
        )
    return ValidatedAnswer(cleaned.strip(), cited)


def guard_generated_answer(answer: str, chunks: list[RetrievedChunk]) -> ValidatedAnswer:
    """Fail closed when generation lacks approved evidence or contains unsafe artifacts."""

    validated = validate_citations(answer, chunks)
    if chunks and not validated.cited_chunks:
        return ValidatedAnswer(UNVERIFIED_MODEL_RESPONSE, [])
    if any(pattern.search(validated.text) for pattern in _UNSAFE_MODEL_PATTERNS):
        return ValidatedAnswer(UNVERIFIED_MODEL_RESPONSE, [])
    return validated


def _is_obviously_off_topic(message: str) -> bool:
    return bool(_OFF_TOPIC_REQUEST.search(message)) and not bool(_HEALTH_CONTEXT.search(message))


class ChatService:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        model: ModelProvider,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.model = model

    def answer(
        self,
        db: Session,
        user: User,
        message: str,
        care_mode: str,
        conversation_id: str | None = None,
    ) -> ChatResponse:
        assessment = assess(message)
        logger.info(
            "chat_assessed user_id=%s fingerprint=%s urgency=%s flags=%s",
            user.id,
            short_fingerprint(message),
            assessment.urgency.value,
            ",".join(assessment.flags) or "none",
        )

        if _SMALL_TALK.fullmatch(message):
            text = (
                "Hello! I’m ready to help you understand a health concern, prepare for a clinical "
                "visit, or review medicine and herb safety. This local experimental mode uses "
                "deterministic mock responses, so it is for testing the workflow—not medical advice."
            )
            return self._response(db, user, message, text, assessment, [], conversation_id)

        if assessment.bypass_model:
            text = emergency_response(
                self.settings.emergency_number,
                self.settings.emergency_region,
                self_harm="self_harm_risk" in assessment.flags,
            )
            return self._response(db, user, message, text, assessment, [], conversation_id)

        if _is_obviously_off_topic(message):
            text = (
                "I am focused on health education, symptom navigation, and medicine or herb safety, "
                "so I cannot complete that unrelated request here. If you have a health-related "
                "question, I can help with that."
            )
            return self._response(db, user, message, text, assessment, [], conversation_id)

        chunks = self.retriever.search(db, message, limit=5)
        if not chunks:
            text = (
                "I could not find an approved, current source for this question, so I cannot answer it "
                "reliably. Please ask a qualified clinician or pharmacist. If symptoms are worsening "
                "or you are worried, seek in-person care."
            )
            return self._response(db, user, message, text, assessment, [], conversation_id)

        prompt = build_user_prompt(message, care_mode, assessment, chunks)
        draft = self.model.generate(SYSTEM_PROMPT, prompt)
        validated = guard_generated_answer(draft, chunks)
        return self._response(
            db,
            user,
            message,
            validated.text,
            assessment,
            validated.cited_chunks,
            conversation_id,
        )

    def _response(
        self,
        db: Session,
        user: User,
        question: str,
        answer: str,
        assessment: SafetyAssessment,
        chunks: list[tuple[str, RetrievedChunk]],
        conversation_id: str | None,
    ) -> ChatResponse:
        saved_id = self._save_history(db, user, question, answer, conversation_id)
        sources = [
            SourceOut(
                id=citation_id,
                title=item.source.title,
                publisher=item.source.publisher,
                url=item.source.url,
                evidence_tier=item.source.evidence_tier,
                reviewed_on=item.source.reviewed_on,
            )
            for citation_id, item in chunks
        ]
        return ChatResponse(
            answer=answer,
            urgency=assessment.urgency.value,
            safety_flags=list(assessment.flags),
            sources=sources,
            disclaimer=DISCLAIMER,
            conversation_id=saved_id,
            history_saved=bool(saved_id),
        )

    def _save_history(
        self,
        db: Session,
        user: User,
        question: str,
        answer: str,
        conversation_id: str | None,
    ) -> str | None:
        if not self.settings.save_chat_history or not self.settings.data_encryption_key:
            return None
        conversation = None
        if conversation_id:
            conversation = db.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user.id,
                )
            )
        if not conversation:
            conversation = Conversation(user_id=user.id, title=question[:117])
            db.add(conversation)
            db.flush()
        cipher = Cipher(self.settings.data_encryption_key)
        db.add_all(
            [
                ChatMessage(
                    conversation_id=conversation.id,
                    role="user",
                    content_ciphertext=cipher.encrypt(question),
                ),
                ChatMessage(
                    conversation_id=conversation.id,
                    role="assistant",
                    content_ciphertext=cipher.encrypt(answer),
                ),
            ]
        )
        conversation.updated_at = utcnow()
        db.commit()
        return conversation.id
