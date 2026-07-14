import json
from pathlib import Path

from app.safety import Urgency, assess


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
