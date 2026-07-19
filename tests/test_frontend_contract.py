from pathlib import Path


def test_frontend_normalizes_structured_api_errors_and_validates_passwords() -> None:
    script = Path("app/static/app.js").read_text(encoding="utf-8")
    assert "function errorMessage(detail)" in script
    assert "item?.msg" in script
    assert "function validateCredentials(email, password, registering)" in script
    assert "Use at least 12 characters" in script
    assert "new Error(body.detail" not in script


def test_frontend_clears_stale_urgent_notice_for_each_answer() -> None:
    script = Path("app/static/app.js").read_text(encoding="utf-8")
    function = script.split("function addAssistantMessage(payload)", 1)[1].split(
        "async function sendMessage", 1
    )[0]
    clear_position = function.index('$("#urgent-note").classList.add("hidden");')
    urgent_branch_position = function.index('if (["emergency", "urgent"].includes(payload.urgency))')
    assert clear_position < urgent_branch_position
