from datetime import date
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

EvidenceStatus = Literal[
    "grounded",
    "safety_bypass",
    "insufficient_sources",
    "model_rejected",
    "not_applicable",
]


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    terms_accepted: bool

    @field_validator("password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        classes = [
            any(char.islower() for char in value),
            any(char.isupper() for char in value),
            any(char.isdigit() for char in value),
        ]
        if sum(classes) < 2:
            raise ValueError("Use at least two of: uppercase, lowercase, and numbers")
        return value

    @model_validator(mode="after")
    def require_terms(self) -> "RegisterRequest":
        if not self.terms_accepted:
            raise ValueError("You must accept the informational-use terms")
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class AccountDeleteRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    id: str
    email: EmailStr
    history_enabled: bool


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=4000)
    care_mode: Literal["biomedical", "integrative"] = "integrative"
    conversation_id: str | None = None
    medical_disclaimer_accepted: Literal[True]

    @field_validator("message")
    @classmethod
    def useful_message(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 2:
            raise ValueError("Please enter a health question")
        return cleaned


class SourceOut(BaseModel):
    id: str
    title: str
    publisher: str
    url: str
    evidence_tier: str
    license: str
    reviewed_on: date


class ChatResponse(BaseModel):
    answer: str
    urgency: Literal["emergency", "urgent", "soon", "routine"]
    safety_flags: list[str]
    evidence_status: EvidenceStatus
    evidence_notice: str
    sources: list[SourceOut]
    disclaimer: str
    conversation_id: str | None = None
    history_saved: bool = False
