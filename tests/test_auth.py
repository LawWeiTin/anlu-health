from fastapi.testclient import TestClient

from tests.conftest import csrf_headers


def test_register_me_logout(client: TestClient) -> None:
    created = client.post(
        "/api/auth/register",
        json={
            "email": "USER@example.com",
            "password": "Strong-passphrase-2026",
            "terms_accepted": True,
        },
    )
    assert created.status_code == 201
    assert created.json()["email"] == "user@example.com"
    assert client.cookies.get("anlu_session")

    me = client.get("/api/me")
    assert me.status_code == 200
    assert me.json()["history_enabled"] is False

    logout = client.post("/api/auth/logout", headers=csrf_headers(client), json={})
    assert logout.status_code == 204
    assert client.get("/api/me").status_code == 401


def test_duplicate_registration_is_rejected(client: TestClient) -> None:
    payload = {
        "email": "same@example.com",
        "password": "Strong-passphrase-2026",
        "terms_accepted": True,
    }
    assert client.post("/api/auth/register", json=payload).status_code == 201
    assert client.post("/api/auth/register", json=payload).status_code == 409


def test_authenticated_write_requires_csrf(registered_client: TestClient) -> None:
    response = registered_client.post(
        "/api/chat", json={"message": "I have a question", "care_mode": "biomedical"}
    )
    assert response.status_code == 403


def test_weak_password_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={"email": "weak@example.com", "password": "aaaaaaaaaaaa", "terms_accepted": True},
    )
    assert response.status_code == 422


def test_metrics_are_available_without_token_only_outside_production(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "anlu_http_requests_total" in response.text
