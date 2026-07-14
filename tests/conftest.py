import os
import tempfile
from pathlib import Path

import pytest

_db_file = Path(tempfile.gettempdir()) / "anlu-tests.sqlite3"
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = f"sqlite:///{_db_file.as_posix()}"
os.environ["COOKIE_SECURE"] = "false"
os.environ["MODEL_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["SAVE_CHAT_HISTORY"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, get_engine  # noqa: E402
from app.main import app  # noqa: E402
from app.rate_limit import get_rate_limiter  # noqa: E402


@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(get_engine())
    Base.metadata.create_all(get_engine())
    get_rate_limiter.cache_clear()
    yield


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def registered_client(client: TestClient) -> TestClient:
    response = client.post(
        "/api/auth/register",
        json={
            "email": "person@example.com",
            "password": "A-safe-password-2026",
            "terms_accepted": True,
        },
    )
    assert response.status_code == 201
    return client


def csrf_headers(client: TestClient) -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies.get("anlu_csrf")}
