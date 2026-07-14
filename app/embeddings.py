import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from app.config import Settings, get_settings


class EmbeddingError(RuntimeError):
    pass


_ENGLISH_STOPWORDS = {
    "a",
    "am",
    "an",
    "and",
    "are",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "or",
    "should",
    "the",
    "these",
    "this",
    "to",
    "what",
    "with",
    "you",
}
_CJK_SEQUENCE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def _feature_tokens(text: str) -> list[str]:
    lowered = text.casefold()
    words = [
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", lowered)
        if len(token) >= 2 and token not in _ENGLISH_STOPWORDS
    ]
    cjk: list[str] = []
    for sequence in _CJK_SEQUENCE.findall(lowered):
        cjk.append(sequence)
        cjk.extend(sequence[index : index + 2] for index in range(max(0, len(sequence) - 1)))
    return words + cjk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class MockMultilingualEmbeddings(EmbeddingProvider):
    """Deterministic local embeddings for tests/UI only; never a production medical retriever."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in _feature_tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                slot = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] % 2 else -1.0
                vector[slot] += sign
            results.append(_normalize(vector))
        return results


class OpenAICompatibleEmbeddings(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        token = self.settings.embedding_api_token or self.settings.hf_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = self.settings.embedding_api_url.rstrip("/")
        if not url.endswith("/embeddings"):
            url += "/embeddings"
        try:
            response = httpx.post(
                url,
                headers=headers,
                json={"model": self.settings.embedding_model, "input": texts},
                timeout=self.settings.model_timeout_seconds,
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in data]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding service unavailable") from exc
        if any(len(vector) != self.settings.embedding_dimensions for vector in vectors):
            raise EmbeddingError("Embedding dimension does not match the database schema")
        return [_normalize(vector) for vector in vectors]


class TEIEmbeddings(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        token = self.settings.embedding_api_token or self.settings.hf_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = httpx.post(
                self.settings.embedding_api_url,
                headers=headers,
                json={"inputs": texts, "normalize": True},
                timeout=self.settings.model_timeout_seconds,
            )
            response.raise_for_status()
            vectors = response.json()
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding service unavailable") from exc
        if not isinstance(vectors, list) or any(
            not isinstance(vector, list) or len(vector) != self.settings.embedding_dimensions
            for vector in vectors
        ):
            raise EmbeddingError("Unexpected embedding response")
        return [_normalize(vector) for vector in vectors]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "openai_compatible":
        return OpenAICompatibleEmbeddings(settings)
    if settings.embedding_provider == "tei":
        return TEIEmbeddings(settings)
    return MockMultilingualEmbeddings(settings.embedding_dimensions)
