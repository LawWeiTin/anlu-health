from pathlib import Path


def test_frontend_normalizes_structured_api_errors_and_validates_passwords() -> None:
    script = Path("app/static/app.js").read_text(encoding="utf-8")
    assert "function errorMessage(detail)" in script
    assert "item?.msg" in script
    assert "function validateCredentials(email, password, registering)" in script
    assert "Use at least 12 characters" in script
    assert "new Error(body.detail" not in script
