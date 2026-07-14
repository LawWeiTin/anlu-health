"""Validate SFT examples and reject obvious identifiers before remote training."""

import argparse
import json
import re
from pathlib import Path

IDENTIFIER_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)"),
    "medical_record_number": re.compile(r"\b(?:mrn|medical record)\s*[:#]?\s*\d{5,}\b", re.I),
}


def validate_example(example: dict[str, object], line_number: int) -> dict[str, object]:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"line {line_number}: messages must contain user and assistant turns")
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles[0] != "user" or roles[-1] != "assistant":
        raise ValueError(f"line {line_number}: expected user ... assistant role order")
    text = "\n".join(
        str(message.get("content", "")) for message in messages if isinstance(message, dict)
    )
    for label, pattern in IDENTIFIER_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"line {line_number}: possible {label} detected")
    assistant = str(messages[-1].get("content", ""))  # type: ignore[union-attr]
    if any(term in assistant.casefold() for term in ("definitely benign", "you have cancer")):
        raise ValueError(f"line {line_number}: prohibited diagnostic certainty")
    if len(text) > 12000:
        raise ValueError(f"line {line_number}: example is unexpectedly long")
    return example


def prepare(source: Path, destination: Path) -> int:
    examples = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            examples.append(validate_example(json.loads(line), line_number))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(example, ensure_ascii=False) for example in examples) + "\n",
        encoding="utf-8",
    )
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"validated {prepare(args.input, args.output)} examples")


if __name__ == "__main__":
    main()
