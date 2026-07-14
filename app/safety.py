import re
from dataclasses import dataclass
from enum import StrEnum


class Urgency(StrEnum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    SOON = "soon"
    ROUTINE = "routine"


@dataclass(frozen=True)
class SafetyAssessment:
    urgency: Urgency
    flags: tuple[str, ...]
    bypass_model: bool = False


_SELF_HARM = re.compile(
    r"\b(kill myself|end my life|suicid(?:e|al)|hurt myself)\b|自杀|结束生命|伤害自己",
    re.IGNORECASE,
)
_DIRECT_EMERGENCY = re.compile(
    r"\b(cannot breathe|can't breathe|severe difficulty breathing|blue lips|"
    r"face (?:is )?droop(?:ing)?|one-sided weakness|uncontrolled bleeding|"
    r"bleeding is uncontrolled|"
    r"passed out|unconscious|"
    r"anaphylaxis|swollen tongue|seizure lasting)\b|"
    r"严重呼吸困难|无法呼吸|嘴唇发蓝|口角歪斜|单侧无力|无法止血|昏迷|严重过敏",
    re.IGNORECASE,
)
_CHEST = re.compile(r"\b(chest pain|chest pressure|crushing chest)\b|胸痛|胸口压迫", re.I)
_BREATH = re.compile(
    r"\b(short(?:ness)? of breath|breathless|difficulty breathing)\b|呼吸困难", re.I
)
_SEVERE_CHEST = re.compile(
    r"\b(sudden|severe|crushing|heavy)\b.{0,24}\b(chest pain|chest pressure)\b|"
    r"\b(chest pain|chest pressure)\b.{0,24}\b(sudden|severe|crushing|heavy)\b|"
    r"突发胸痛|剧烈胸痛|胸口剧烈压迫",
    re.I,
)
_OTHER_EMERGENCY = re.compile(
    r"\b(pulsating abdominal lump|pregnan\w* .{0,20} heavy bleeding|"
    r"sudden testicular pain|stiff neck .{0,20} fever)\b|"
    r"腹部搏动性肿块|孕期大量出血|突发睾丸疼痛|发烧.{0,10}颈部僵硬",
    re.I,
)
_LUMP = re.compile(r"\b(lump|mass|swelling|nodule|bump)\b|肿块|包块|结节|肿胀", re.I)
_LUMP_URGENT = re.compile(
    r"\b(rapidly growing|growing quickly|hard and fixed|fixed lump|testicle|testicular|"
    r"breast lump|trouble swallowing|difficulty swallowing|red and hot|pus|high fever)\b|"
    r"快速长大|质硬固定|睾丸|乳房肿块|吞咽困难|红肿发热|流脓|高烧",
    re.I,
)
_LUMP_SOON = re.compile(
    r"\b(two weeks|2 weeks|persistent|getting bigger|weight loss|night sweats|recurring)\b|"
    r"两周|持续存在|越来越大|体重下降|盗汗|反复出现",
    re.I,
)


def assess(message: str) -> SafetyAssessment:
    flags: list[str] = []

    if _SELF_HARM.search(message):
        return SafetyAssessment(Urgency.EMERGENCY, ("self_harm_risk",), True)
    if _DIRECT_EMERGENCY.search(message):
        return SafetyAssessment(Urgency.EMERGENCY, ("emergency_red_flag",), True)
    if _SEVERE_CHEST.search(message) or _OTHER_EMERGENCY.search(message):
        return SafetyAssessment(Urgency.EMERGENCY, ("emergency_red_flag",), True)
    if _CHEST.search(message) and _BREATH.search(message):
        return SafetyAssessment(Urgency.EMERGENCY, ("cardiorespiratory_red_flag",), True)
    if _CHEST.search(message):
        return SafetyAssessment(Urgency.URGENT, ("chest_symptom",))

    if _LUMP.search(message):
        flags.append("lump_or_swelling")
        if _LUMP_URGENT.search(message):
            flags.append("concerning_lump_feature")
            return SafetyAssessment(Urgency.URGENT, tuple(flags))
        if _LUMP_SOON.search(message):
            flags.append("persistent_lump_feature")
        return SafetyAssessment(Urgency.SOON, tuple(flags))

    return SafetyAssessment(Urgency.ROUTINE, tuple(flags))


def emergency_response(number: str, region: str, self_harm: bool = False) -> str:
    if self_harm:
        return (
            "Your message may describe an immediate risk of self-harm. Please move away from anything "
            "you could use to hurt yourself, stay with another person if possible, and call your local "
            f"emergency service now ({number} in {region}). If calling is difficult, ask someone nearby "
            "to call or go to the nearest emergency department. I cannot safely handle this by chat."
        )
    return (
        "This could be an emergency. Call your local emergency service now "
        f"({number} in {region}) or go to the nearest emergency department. Do not drive yourself. "
        "If you are with the person, keep them safe and follow the dispatcher's instructions. "
        "I cannot safely assess this through chat."
    )
