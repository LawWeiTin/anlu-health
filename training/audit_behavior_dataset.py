"""Reproducible quality and leakage audit for the project-authored SFT behavior set."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from training.prepare_open_datasets import IDENTIFIER_PATTERNS, PROHIBITED_CERTAINTY

REQUIRED_TAG_COUNTS = {
    "smalltalk": 4,
    "hemoptysis": 5,
    "lump": 9,
    "tcm": 14,
    "interaction": 7,
    "topic_relevance": 7,
    "citation_integrity": 5,
    "emergency": 4,
    "zh": 4,
    "no_dosing": 4,
    "pregnancy": 3,
}
SPECIAL_EVIDENCE_KEYS = {"project-safety-policy"}
MAX_ASSISTANT_WORDS = 180


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9\u3400-\u9fff]+", text.casefold()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(
    behavior_path: Path,
    source_registry_path: Path,
    heldout_paths: list[Path],
) -> dict[str, Any]:
    records = _read_jsonl(behavior_path)
    registry = yaml.safe_load(source_registry_path.read_text(encoding="utf-8"))
    approved_sources = {
        item["source_key"] for item in registry["sources"] if item.get("approved")
    }
    scenario_ids: set[str] = set()
    prompts: dict[str, str] = {}
    tag_counts: Counter[str] = Counter()
    failures: list[str] = []

    for line_number, record in enumerate(records, start=1):
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            failures.append(f"line {line_number}: missing scenario_id")
        elif scenario_id in scenario_ids:
            failures.append(f"line {line_number}: duplicate scenario_id {scenario_id}")
        else:
            scenario_ids.add(scenario_id)

        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            failures.append(f"line {line_number}: expected one user and one assistant message")
            continue
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            failures.append(f"line {line_number}: invalid role order")
            continue
        prompt = str(messages[0].get("content", "")).strip()
        assistant = str(messages[1].get("content", "")).strip()
        normalized = _normalize(prompt)
        if not normalized or not assistant:
            failures.append(f"line {line_number}: empty prompt or answer")
        if len(assistant.split()) > MAX_ASSISTANT_WORDS:
            failures.append(
                f"line {line_number}: assistant response exceeds {MAX_ASSISTANT_WORDS} words"
            )
        if normalized in prompts:
            failures.append(
                f"line {line_number}: duplicate normalized prompt shared with {prompts[normalized]}"
            )
        prompts[normalized] = str(scenario_id)
        combined = f"{prompt}\n{assistant}"
        for label, pattern in IDENTIFIER_PATTERNS.items():
            if pattern.search(combined):
                failures.append(f"line {line_number}: possible {label}")
        if any(term in assistant.casefold() for term in PROHIBITED_CERTAINTY):
            failures.append(f"line {line_number}: prohibited diagnostic certainty")

        tags = record.get("tags")
        if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) for tag in tags):
            failures.append(f"line {line_number}: tags must be a non-empty string list")
            tags = []
        tag_counts.update(tags)
        evidence_keys = record.get("evidence_source_keys")
        if not isinstance(evidence_keys, list) or any(
            not isinstance(key, str) for key in evidence_keys
        ):
            failures.append(f"line {line_number}: invalid evidence_source_keys")
            evidence_keys = []
        unknown = set(evidence_keys) - approved_sources - SPECIAL_EVIDENCE_KEYS
        if unknown:
            failures.append(f"line {line_number}: unknown evidence keys {sorted(unknown)}")
        if "smalltalk" in tags and "emergency service" in assistant.casefold():
            failures.append(f"line {line_number}: smalltalk answer forces emergency boilerplate")
        if "no_dosing" in tags and not any(
            phrase in assistant.casefold()
            for phrase in ("cannot provide", "cannot choose", "cannot recommend")
        ):
            failures.append(f"line {line_number}: no_dosing answer lacks an explicit refusal")

    for tag, minimum in REQUIRED_TAG_COUNTS.items():
        if tag_counts[tag] < minimum:
            failures.append(f"coverage: {tag} has {tag_counts[tag]} examples; need {minimum}")

    heldout_prompts: list[tuple[str, str]] = []
    for path in heldout_paths:
        for record in _read_jsonl(path):
            text = str(record.get("message") or record.get("prompt") or "")
            heldout_prompts.append((str(record.get("id", path.name)), _normalize(text)))
    leakage: list[dict[str, str | float]] = []
    for train_prompt, scenario_id in prompts.items():
        for heldout_id, heldout_prompt in heldout_prompts:
            similarity = SequenceMatcher(None, train_prompt, heldout_prompt).ratio()
            if train_prompt == heldout_prompt or similarity >= 0.92:
                leakage.append(
                    {
                        "scenario_id": scenario_id,
                        "heldout_id": heldout_id,
                        "similarity": round(similarity, 4),
                    }
                )
    if leakage:
        failures.append(f"held-out prompt leakage: {len(leakage)} exact/near-duplicate pairs")

    source_counts = Counter(
        key for record in records for key in record.get("evidence_source_keys", [])
    )
    return {
        "status": "pass" if not failures else "fail",
        "records": len(records),
        "unique_scenarios": len(scenario_ids),
        "tag_counts": dict(sorted(tag_counts.items())),
        "evidence_source_counts": dict(sorted(source_counts.items())),
        "heldout_prompts_checked": len(heldout_prompts),
        "leakage": leakage,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", type=Path, default=Path("training/data/sample_sft.jsonl"))
    parser.add_argument("--registry", type=Path, default=Path("data/source_registry.yaml"))
    parser.add_argument(
        "--heldout",
        type=Path,
        action="append",
        default=[
            Path("data/eval/golden_cases.jsonl"),
            Path("training/data/model_release_cases.jsonl"),
        ],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.behavior, args.registry, args.heldout)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
