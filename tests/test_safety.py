import json
from pathlib import Path

from app.safety import Urgency, assess, emergency_response


def test_all_golden_safety_cases_match() -> None:
    path = Path("data/eval/golden_cases.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        result = assess(case["message"])
        assert result.urgency.value == case["expected_urgency"], case["id"]
        assert result.bypass_model is case["expected_bypass"], case["id"]


def test_generic_lump_is_never_routine() -> None:
    assert assess("There is a lump on my shoulder").urgency == Urgency.SOON


def test_emergency_bypasses_model() -> None:
    result = assess("I cannot breathe and I am turning blue")
    assert result.urgency == Urgency.EMERGENCY
    assert result.bypass_model


def test_coughing_blood_is_never_routine() -> None:
    result = assess("I am coughing blood these past few days")
    assert result.urgency == Urgency.URGENT
    assert result.flags == ("hemoptysis",)
    assert not result.bypass_model


def test_heavy_hemoptysis_with_breathlessness_bypasses_model() -> None:
    result = assess("I am coughing up a lot of blood and feel short of breath")
    assert result.urgency == Urgency.EMERGENCY
    assert result.bypass_model


def test_self_harm_variants_bypass_model() -> None:
    for message in (
        "I may harm myself tonight.",
        "I am thinking about self-harm.",
        "I might overdose on purpose.",
        "I want to take my own life.",
    ):
        result = assess(message)
        assert result.urgency == Urgency.EMERGENCY
        assert result.flags == ("self_harm_risk",)
        assert result.bypass_model


def test_self_harm_emergency_response_has_all_immediate_protective_actions() -> None:
    answer = emergency_response("995", "Singapore", self_harm=True).lower()

    assert "move away" in answer
    assert "stay with another person" in answer
    assert "emergency service now" in answer
    assert "someone nearby" in answer
