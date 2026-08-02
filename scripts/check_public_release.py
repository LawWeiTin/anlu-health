"""Fail closed when tracked files contain credentials or private service metadata."""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DOTENV = {".env.example"}
PATTERNS = {
    "Hugging Face access token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub access token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "OpenAI API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "JWT-like bearer token": re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    ),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential-bearing URL": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    "private Hugging Face endpoint": re.compile(
        r"https://[^\s\"']+\.endpoints\.huggingface\.cloud"
    ),
    "private Kaggle notebook URL": re.compile(r"https://www\.kaggle\.com/code/"),
    "Google Drive URL": re.compile(r"https://drive\.google\.com/"),
    "personal Gmail address": re.compile(r"\b[A-Za-z0-9._%+-]+@gmail\.com\b", re.IGNORECASE),
    "Windows user path": re.compile(r"\b[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "Unix home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
}


def _tracked_files() -> list[Path]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for the public release audit")
    result = subprocess.run(  # noqa: S603
        [git, "ls-files", "-z"],  # nosec B603
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def audit() -> list[str]:
    findings: list[str] = []
    for path in _tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        name = path.name.lower()
        if name.startswith(".env") and name not in ALLOWED_DOTENV:
            findings.append(f"tracked dotenv file: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{label}: {relative}")
    return sorted(set(findings))


def main() -> None:
    findings = audit()
    if findings:
        print("public_release_gate_failed")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print("public_release_gate_passed")


if __name__ == "__main__":
    main()
