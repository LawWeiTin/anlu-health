import pytest
from pydantic import ValidationError

from app.config import Settings


def production_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_env": "production",
        "cookie_secure": True,
        "model_provider": "openai_compatible",
        "model_api_url": "https://models.example/v1",
        "embedding_provider": "tei",
        "embedding_api_url": "https://embeddings.example",
        "hf_token": "test-inference-token",
        "metrics_token": "test-metrics-token",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_production_configuration_accepts_secure_dependencies() -> None:
    assert production_settings().app_env == "production"


def test_production_requires_private_metrics_token() -> None:
    with pytest.raises(ValidationError, match="METRICS_TOKEN"):
        production_settings(metrics_token=None)
