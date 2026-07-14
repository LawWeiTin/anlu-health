from fastapi.testclient import TestClient


def test_ready_identifies_offline_experimental_runtime(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["runtime_mode"] == "local_experimental"
    assert response.json()["history_storage"] == "disabled"
