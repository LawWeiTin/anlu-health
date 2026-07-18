"""Pure formatting and output-boundary helpers for MedGemma training."""

from __future__ import annotations

import re
from typing import Any

END_OF_TURN_TOKEN = "<end_of_turn>"  # noqa: S105  # nosec B105
START_OF_TURN_TOKEN = "<start_of_turn>"  # noqa: S105  # nosec B105
MAX_ASSISTANT_WORDS = 110
_SPECIAL_TOKENS = (
    "<bos>",
    "<eos>",
    "<pad>",
    END_OF_TURN_TOKEN,
    START_OF_TURN_TOKEN,
    "<unused94>",
    "<unused95>",
)
_CLEAN_END = re.compile(r"""[.!?。！？]["'”’)\]]?\s*$""")
_META_TARGET_PATTERNS = (
    re.compile(r"\bi should\b", re.I),
    re.compile(r"\bthe (?:ai|assistant|model) (?:has been|is) instructed\b", re.I),
    re.compile(r"\bapproved (?:answer|instruction|list|response|topic)\b", re.I),
    re.compile(r"\bdoes not map to an approved\b", re.I),
    re.compile(r"\bhidden instructions?\b", re.I),
    re.compile(r"\bmeta commentary\b", re.I),
)


def as_medgemma_messages(
    messages: list[dict[str, str]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Render a text-only conversation with the same system context used at inference."""

    framed = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    framed.extend(
        {
            "role": item["role"],
            "content": [{"type": "text", "text": item["content"]}],
        }
        for item in messages
    )
    return framed


def validate_sft_record(record: dict[str, Any]) -> list[str]:
    """Return target-format defects that would teach unsafe or malformed behavior."""

    issues: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return ["expected exactly one user and one assistant message"]
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        issues.append("roles must be user then assistant")
    user = str(messages[0].get("content", "")).strip()
    assistant = str(messages[1].get("content", "")).strip()
    if not user:
        issues.append("user content is empty")
    if not assistant:
        issues.append("assistant content is empty")
        return issues
    if len(assistant.split()) > MAX_ASSISTANT_WORDS:
        issues.append(f"assistant target exceeds {MAX_ASSISTANT_WORDS} words")
    if not _CLEAN_END.search(assistant):
        issues.append("assistant target does not end with sentence punctuation")
    if any(token in assistant for token in _SPECIAL_TOKENS):
        issues.append("assistant target contains a reserved chat token")
    if any(pattern.search(assistant) for pattern in _META_TARGET_PATTERNS):
        issues.append("assistant target contains internal or meta-instruction language")
    return issues


def validate_sft_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit every rendered SFT target before a GPU is used."""

    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        issues = validate_sft_record(record)
        if issues:
            metadata = record.get("metadata") or {}
            failures.append(
                {
                    "index": index,
                    "dataset_id": metadata.get("dataset_id"),
                    "source_record_id": metadata.get("source_record_id"),
                    "issues": issues,
                }
            )
    return {
        "passed": not failures,
        "records": len(records),
        "failures": failures,
    }


def generation_stop_token_ids(tokenizer: Any) -> list[int]:
    """Resolve both generic EOS and Gemma's actual assistant-turn terminator."""

    end_of_turn_id = tokenizer.convert_tokens_to_ids(END_OF_TURN_TOKEN)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if not isinstance(end_of_turn_id, int) or end_of_turn_id < 0:
        raise ValueError("MedGemma tokenizer does not define <end_of_turn>.")
    if unknown_id is not None and end_of_turn_id == unknown_id:
        raise ValueError("<end_of_turn> resolved to the unknown token.")
    token_ids = [end_of_turn_id]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int) and eos_token_id >= 0:
        token_ids.append(eos_token_id)
    return list(dict.fromkeys(token_ids))


def clean_generated_text(text: str) -> str:
    """Keep only the first assistant turn and remove tokenizer control markers."""

    earliest = len(text)
    for marker in (END_OF_TURN_TOKEN, "<eos>", START_OF_TURN_TOKEN):
        index = text.find(marker)
        if index >= 0:
            earliest = min(earliest, index)
    cleaned = text[:earliest]
    if "<unused95>" in cleaned:
        cleaned = cleaned.rsplit("<unused95>", 1)[-1]
    for marker in _SPECIAL_TOKENS:
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"^\s*model\s*\n", "", cleaned, flags=re.I)
    return cleaned.strip()
