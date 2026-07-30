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


def test_frontend_renders_explicit_evidence_state_and_fail_closed_fallback() -> None:
    script = Path("app/static/app.js").read_text(encoding="utf-8")
    function = script.split("function addAssistantMessage(payload)", 1)[1].split(
        "async function sendMessage", 1
    )[0]

    assert 'payload.evidence_status || "model_rejected"' in function
    assert 'evidenceLabel.textContent = "Evidence check"' in function
    assert "payload.evidence_notice" in function
    assert 'evidence_status: "model_rejected"' in script
    assert 'evidence_notice: "No verified response was produced."' in script


def test_frontend_requires_explicit_medical_acknowledgement_and_renders_detail_headings() -> None:
    page = Path("app/static/index.html").read_text(encoding="utf-8")
    script = Path("app/static/app.js").read_text(encoding="utf-8")

    assert 'id="medical-disclaimer"' in page
    assert "educational possibilities and examples" in page
    assert "medical_disclaimer_accepted: true" in script
    assert 'if (!$("#medical-disclaimer").checked)' in script
    assert '$("#medical-disclaimer").checked = false' in script
    assert '"Possible explanations"' in script
    assert '"Concrete examples"' in script
    assert "Open Government Licence v3.0" in script


def test_frontend_wraps_long_grounded_answers_on_narrow_screens() -> None:
    styles = Path("app/static/styles.css").read_text(encoding="utf-8")

    assert ".assistant-message > div:last-child { min-width: 0; }" in styles
    assert ".answer-body { overflow-wrap: anywhere;" in styles
    assert ".source-card > span:first-child { min-width: 0; overflow-wrap: anywhere; }" in styles
