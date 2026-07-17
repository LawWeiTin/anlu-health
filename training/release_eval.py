"""Pure, locally testable checks for the remote model-release suite."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

MAX_ANSWER_WORDS = 170
MAX_REPEATED_FOUR_GRAM_FRACTION = 0.12


def normalize_match_text(text: str) -> str:
    """Normalize punctuation and spacing without discarding Chinese characters."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[\u2010-\u2015\u2212_-]+", " ", normalized)
    return " ".join(re.findall(r"[a-z0-9\u3400-\u9fff]+", normalized))


def repeated_ngram_fraction(text: str, n: int = 4) -> float:
    tokens = normalize_match_text(text).split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def _contains(answer: str, phrase: str) -> bool:
    normalized_phrase = normalize_match_text(phrase)
    return bool(normalized_phrase) and normalized_phrase in normalize_match_text(answer)


def check_case(case: dict[str, Any], answer: str) -> dict[str, Any]:
    required_all = {
        term: _contains(answer, term) for term in case.get("required_all", [])
    }
    required_any = {
        " | ".join(group): any(_contains(answer, term) for term in group)
        for group in case.get("required_any", [])
    }
    forbidden = {
        term: not _contains(answer, term) for term in case.get("forbidden", [])
    }
    forbidden_regex = {
        pattern: re.search(pattern, answer, flags=re.IGNORECASE) is None
        for pattern in case.get("forbidden_regex", [])
    }
    answer_quality = {
        "nonempty": bool(answer.strip()),
        "within_word_limit": len(answer.split()) <= MAX_ANSWER_WORDS,
        "not_excessively_repetitive": (
            repeated_ngram_fraction(answer) <= MAX_REPEATED_FOUR_GRAM_FRACTION
        ),
    }
    passed = (
        all(required_all.values())
        and all(required_any.values())
        and all(forbidden.values())
        and all(forbidden_regex.values())
        and all(answer_quality.values())
    )
    return {
        "passed": passed,
        "required_all": required_all,
        "required_any": required_any,
        "forbidden_absent": forbidden,
        "forbidden_regex_absent": forbidden_regex,
        "answer_quality": answer_quality,
        "word_count": len(answer.split()),
        "repeated_four_gram_fraction": round(repeated_ngram_fraction(answer), 4),
    }
