"""Shared text preparation for semantic and lexical knowledge retrieval."""

from __future__ import annotations

from collections.abc import Iterable

_ENGLISH_STOPWORDS = {
    "a",
    "about",
    "am",
    "an",
    "and",
    "answer",
    "are",
    "article",
    "been",
    "do",
    "document",
    "evidence",
    "few",
    "for",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "or",
    "question",
    "retrieved",
    "should",
    "source",
    "supplied",
    "the",
    "these",
    "this",
    "to",
    "use",
    "what",
    "with",
    "you",
}


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _word_runs(text: str) -> list[str]:
    """Split Unicode text without regular-expression topic or phrase rules."""

    runs: list[str] = []
    current: list[str] = []
    current_is_cjk: bool | None = None
    for character in text.casefold():
        if not (character.isalnum() or _is_cjk(character)):
            if current:
                runs.append("".join(current))
                current = []
                current_is_cjk = None
            continue
        is_cjk = _is_cjk(character)
        if current and is_cjk != current_is_cjk:
            runs.append("".join(current))
            current = []
        current.append(character)
        current_is_cjk = is_cjk
    if current:
        runs.append("".join(current))
    return runs


def search_tokens(text: str, *, include_bigrams: bool = True) -> list[str]:
    """Return auditable lexical features for BM25 and database full-text search."""

    unigrams: list[str] = []
    for run in _word_runs(text):
        if all(_is_cjk(character) for character in run):
            if len(run) == 1:
                unigrams.append(run)
            else:
                unigrams.extend(run[index : index + 2] for index in range(len(run) - 1))
            continue
        if len(run) >= 2 and run not in _ENGLISH_STOPWORDS:
            unigrams.append(run)

    if not include_bigrams:
        return unigrams
    bigrams = [
        f"{left}_{right}"
        for left, right in zip(unigrams, unigrams[1:], strict=False)
        if left != right
    ]
    return unigrams + bigrams


def knowledge_search_text(
    *,
    title: str,
    publisher: str,
    topics: Iterable[str],
    keywords: Iterable[str] = (),
    content: str,
) -> str:
    """Build the stored lexical document from visible evidence and declared metadata."""

    topic_text = " ".join(str(topic).replace("_", " ") for topic in topics)
    keyword_text = " ".join(str(keyword) for keyword in keywords)
    base = " ".join(
        part.strip()
        for part in (title, title, publisher, topic_text, keyword_text, content)
        if part
    )
    features = search_tokens(base)
    return f"{base}\n{' '.join(features)}"


def semantic_document_text(
    *,
    title: str,
    publisher: str,
    topics: Iterable[str],
    keywords: Iterable[str] = (),
    content: str,
) -> str:
    """Format a passage for E5-style and compatible embedding services."""

    topic_text = ", ".join(str(topic).replace("_", " ") for topic in topics)
    keyword_text = ", ".join(str(keyword) for keyword in keywords)
    return (
        f"passage: {title}. Publisher: {publisher}. Topics: {topic_text}. "
        f"Index terms: {keyword_text}. "
        f"Evidence: {content}"
    )


def semantic_query_text(query: str) -> str:
    return f"query: {query.strip()}"


def retrieval_query_text(message: str) -> str:
    """Prefer a sufficiently specific final clause in multi-clause user messages."""

    clauses: list[str] = []
    current: list[str] = []
    for character in message:
        if character in ".?!。！？,，;；":
            clause = "".join(current).strip()
            if clause:
                clauses.append(clause)
            current = []
        else:
            current.append(character)
    final_clause = "".join(current).strip()
    if final_clause:
        clauses.append(final_clause)
    if len(clauses) < 2:
        return message.strip()
    candidate = clauses[-1]
    if len(set(search_tokens(candidate, include_bigrams=False))) >= 3:
        return candidate
    return message.strip()
