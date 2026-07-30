"""Create a license-audited SFT pilot bundle in Colab or another remote runtime.

The command intentionally clones pinned public sources at runtime. Do not run it on the laptop;
the Colab notebook points its work directory at ephemeral `/content` storage and writes only a
new timestamped bundle to private Google Drive.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess  # nosec B404
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from defusedxml import ElementTree

IDENTIFIER_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)"),
    "medical_record_number": re.compile(r"\b(?:mrn|medical record)\s*[:#]?\s*\d{5,}\b", re.I),
}
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
DOSING_PATTERN = re.compile(
    r"\b(?:take|dose|dosage|administer)\b.{0,40}\b\d+(?:\.\d+)?\s*"
    r"(?:mg|mcg|g|ml|milligrams?|micrograms?|tablets?|capsules?)\b",
    re.I,
)
PROHIBITED_CERTAINTY = ("definitely benign", "you have cancer", "this proves you have")
MEDQUAD_MAX_ANSWER_WORDS = 80
PUBMEDQA_MAX_ANSWER_WORDS = 75
SENTENCE_TERMINAL_PATTERN = re.compile(r"""[.!?。！？]["')\]]?$""")


def normalize_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def truncate_to_complete_sentences(text: str, max_words: int) -> str:
    """Keep source text within budget and normalize a missing terminal punctuation mark."""

    words = text.split()
    if len(words) <= max_words:
        result = text.strip()
    else:
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        kept: list[str] = []
        word_count = 0
        for sentence in sentences:
            sentence_words = sentence.split()
            if not sentence_words or word_count + len(sentence_words) > max_words:
                break
            kept.append(sentence)
            word_count += len(sentence_words)
        result = " ".join(kept).strip()

    if result and not SENTENCE_TERMINAL_PATTERN.search(result):
        result += "."
    return result


def _record_is_safe(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if not roles or roles[0] != "user" or roles[-1] != "assistant":
        return False
    combined = "\n".join(
        str(message.get("content", "")) for message in messages if isinstance(message, dict)
    )
    if not 20 <= len(combined) <= 12000:
        return False
    if any(pattern.search(combined) for pattern in IDENTIFIER_PATTERNS.values()):
        return False
    assistant = str(messages[-1].get("content", "")).casefold()
    return not any(term in assistant for term in PROHIBITED_CERTAINTY)


def _sample(records: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> str:
        metadata = record["metadata"]
        value = f"{seed}:{metadata['dataset_id']}:{metadata['source_record_id']}"
        return hashlib.sha256(value.encode()).hexdigest()

    return sorted(records, key=key)[:maximum]


def _dataset_metadata(spec: dict[str, Any], source_record_id: str, **extra: object) -> dict:
    return {
        "dataset_id": spec["id"],
        "source_record_id": source_record_id,
        "source_revision": spec["revision"],
        "license": spec["license"],
        "dataset_homepage": spec["homepage"],
        **extra,
    }


def load_medquad(repo: Path, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    allowed_directories = set(spec["allowed_directories"])
    allowed_types = {item.casefold() for item in spec["allowed_question_types"]}
    records: list[dict[str, Any]] = []
    counters: Counter = Counter()

    for path in sorted(repo.rglob("*.xml")):
        relative = path.relative_to(repo)
        if not relative.parts or relative.parts[0] not in allowed_directories:
            counters["directory_excluded"] += 1
            continue
        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError:
            counters["invalid_xml"] += 1
            continue
        source_url = normalize_text(root.get("url"))
        focus = normalize_text(root.findtext("Focus"))
        for pair in root.findall(".//QAPair"):
            question_element = pair.find("Question")
            answer_element = pair.find("Answer")
            question_type = normalize_text(
                question_element.get("qtype") if question_element is not None else ""
            ).casefold()
            if question_type not in allowed_types:
                counters["question_type_excluded"] += 1
                continue
            question = normalize_text(question_element.text if question_element is not None else "")
            answer = normalize_text(answer_element.text if answer_element is not None else "")
            answer = truncate_to_complete_sentences(answer, MEDQUAD_MAX_ANSWER_WORDS)
            if len(question) < 8 or len(answer) < 40:
                counters["missing_or_short"] += 1
                continue
            if DOSING_PATTERN.search(answer):
                counters["dosing_excluded"] += 1
                continue
            source_id = (
                normalize_text(question_element.get("qid") if question_element is not None else "")
                or f"{relative.as_posix()}:{pair.get('pid', 'unknown')}"
            )
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Provide general educational information, not a personal diagnosis. "
                            + question
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"General educational information: {answer}\n\n"
                            "This describes general medical information and does not diagnose an individual."
                        ),
                    },
                ],
                "metadata": _dataset_metadata(
                    spec,
                    source_id,
                    task="consumer_health_education",
                    question_type=question_type,
                    source_url=source_url,
                    focus=focus,
                    group_id=f"medquad:{focus.casefold() or source_id}",
                ),
            }
            if _record_is_safe(record):
                records.append(record)
            else:
                counters["safety_filter_excluded"] += 1
    counters["accepted_before_sampling"] = len(records)
    return records, counters


def load_pubmedqa(repo: Path, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    data = json.loads((repo / "data" / "ori_pqal.json").read_text(encoding="utf-8"))
    held_out = set(
        json.loads((repo / "data" / "test_ground_truth.json").read_text(encoding="utf-8"))
    )
    records: list[dict[str, Any]] = []
    counters: Counter = Counter({"official_test_ids_reserved": len(held_out)})

    for pmid, example in sorted(data.items()):
        if pmid in held_out:
            counters["official_test_examples_excluded"] += 1
            continue
        question = normalize_text(example.get("QUESTION"))
        contexts = [normalize_text(item) for item in example.get("CONTEXTS", [])]
        context = " ".join(item for item in contexts if item)
        long_answer = normalize_text(example.get("LONG_ANSWER"))
        long_answer = truncate_to_complete_sentences(long_answer, PUBMEDQA_MAX_ANSWER_WORDS)
        decision = normalize_text(example.get("final_decision")).casefold()
        if decision not in {"yes", "no", "maybe"} or not question or not long_answer or not context:
            counters["incomplete_example"] += 1
            continue
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Answer this biomedical research question using only the supplied abstract. "
                        "State yes, no, or maybe, then briefly explain the evidence and uncertainty.\n\n"
                        f"Question: {question}\n\nAbstract: {context}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"Evidence conclusion: {decision.capitalize()}. {long_answer}",
                },
            ],
            "metadata": _dataset_metadata(
                spec,
                pmid,
                task="evidence_conditioned_biomedical_reasoning",
                pmid=pmid,
                decision=decision,
                group_id=f"pubmedqa:{pmid}",
            ),
        }
        if _record_is_safe(record):
            records.append(record)
        else:
            counters["safety_filter_excluded"] += 1
    counters["accepted_before_sampling"] = len(records)
    return records, counters


def load_project_behavior(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_revision = f"sha256:{_sha256(path)}"
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        source = json.loads(line)
        tags = [str(tag) for tag in source.get("tags", [])]
        record = {
            "messages": source["messages"],
            "metadata": {
                "dataset_id": "anlu-authored-safety",
                "source_record_id": source["scenario_id"],
                "source_revision": source_revision,
                "license": "project-authored",
                "dataset_homepage": "private-repository",
                "task": "safety_behavior",
                "tags": tags,
                "evidence_source_keys": source.get("evidence_source_keys", []),
                "authoring_status": "project-authored; clinical review pending",
                "group_id": f"anlu:{source['scenario_id']}",
                "force_split": "train",
            },
        }
        if not _record_is_safe(record):
            raise ValueError(f"project behavior example {line_number} failed validation")
        records.append(record)
    return records


def split_by_group(
    records: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    threshold = int(validation_fraction * 10_000)
    for record in records:
        metadata = record["metadata"]
        if metadata.get("force_split") == "train":
            train.append(record)
            continue
        digest = hashlib.sha256(f"{seed}:{metadata['group_id']}".encode()).hexdigest()
        destination = validation if int(digest[:8], 16) % 10_000 < threshold else train
        destination.append(record)
    return train, validation


def _clone_pinned(spec: dict[str, Any], work_dir: Path) -> Path:
    destination = work_dir / spec["id"]
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing source directory: {destination}")
    destination.mkdir(parents=True)
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is required in the remote training runtime")
    commands = [
        [git, "-C", str(destination), "init", "--quiet"],
        [git, "-C", str(destination), "remote", "add", "origin", spec["repository"]],
        [
            git,
            "-C",
            str(destination),
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "origin",
            spec["revision"],
        ],
        [git, "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
    ]
    for command in commands:
        subprocess.run(command, check=True)  # noqa: S603  # nosec B603
    actual = subprocess.run(  # noqa: S603  # nosec B603
        [git, "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != spec["revision"]:
        raise RuntimeError(f"revision mismatch for {spec['id']}")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def prepare_bundle(
    manifest_path: Path,
    behavior_path: Path,
    output_dir: Path,
    work_dir: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("storage_policy") != "remote_colab_and_private_drive_only":
        raise ValueError("open-dataset manifest must retain the remote-only storage policy")
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    seed = int(manifest["seed"])
    dataset_specs = {item["id"]: item for item in manifest["datasets"]}
    records: list[dict[str, Any]] = []
    audit: dict[str, dict[str, int]] = {}

    for dataset_id, loader in (("medquad", load_medquad), ("pubmedqa", load_pubmedqa)):
        spec = dataset_specs[dataset_id]
        repo = source_root / dataset_id if source_root else _clone_pinned(spec, work_dir)
        loaded, counters = loader(repo, spec)
        sampled = _sample(loaded, int(spec["max_examples"]), seed)
        records.extend(sampled)
        counters["selected_for_pilot"] = len(sampled)
        audit[dataset_id] = dict(counters)

    behavior = load_project_behavior(behavior_path)
    records.extend(behavior)
    audit["anlu-authored-safety"] = {"selected_for_pilot": len(behavior)}

    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        question = normalize_text(record["messages"][0]["content"]).casefold()
        unique.setdefault(hashlib.sha256(question.encode()).hexdigest(), record)
    deduplicated = list(unique.values())
    train, validation = split_by_group(deduplicated, float(manifest["validation_fraction"]), seed)
    if not train or not validation:
        raise ValueError("both train and validation splits must contain records")
    train_groups = {item["metadata"]["group_id"] for item in train}
    validation_groups = {item["metadata"]["group_id"] for item in validation}
    if train_groups & validation_groups:
        raise ValueError("group leakage detected between train and validation")

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)
    bundle_manifest = {
        "status": "unreviewed_pilot_dataset",
        "created_at": datetime.now(UTC).isoformat(),
        "source_manifest": manifest,
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "deduplicated_total": len(deduplicated),
        },
        "audit": audit,
        "artifacts": {
            "train.jsonl": _sha256(train_path),
            "validation.jsonl": _sha256(validation_path),
        },
        "privacy": {
            "contains_user_conversations": False,
            "obvious_identifier_filter_applied": True,
        },
        "training_sampling": {
            "behavior_sampling_weight": int(manifest["behavior_sampling_weight"]),
        },
        "promotion_allowed": False,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    attribution = """# Dataset attribution

- MedQuAD, CC BY 4.0: https://github.com/abachaa/MedQuAD
- PubMedQA, MIT: https://github.com/pubmedqa/pubmedqa
- Anlu reviewed safety examples: project-authored synthetic examples; no user conversations.

MedMCQA is registered as evaluation-only and is not included in this training bundle. The dataset
manifest records exact source revisions, selection counts, exclusions, and artifact checksums.
"""
    (output_dir / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
    return bundle_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--behavior-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Use pre-populated medquad/pubmedqa directories; intended only for isolated tests.",
    )
    args = parser.parse_args()
    result = prepare_bundle(
        args.manifest,
        args.behavior_data,
        args.output_dir,
        args.work_dir,
        args.source_root,
    )
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
