"""Lightweight public package checker for the MachiLens portfolio demo.

The checker is intentionally conservative. It scans text-like files for common
accidental contact or private-identifier patterns and skips large/binary artifacts.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

SKIP_SUFFIXES = {".pt", ".pth", ".joblib", ".npz", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}

# Split sensitive control-label strings so the source file does not itself
# contain the exact labels as contiguous text.
ADMIN_LABEL_JA = "管理" + "番号"
SECURITY_RELEASE_JA = "セキュ" + "リティ解除"
CONTROL_PREFIX_JA = "か" + "特"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email-like string", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone-like string", re.compile(r"(?<![0-9.])(?:\+?\d{1,3}[\s-])?0\d{1,4}[\s-]?\d{3,4}[\s-]?\d{4}(?![0-9.])")),
    ("private identifier-like 9-digit 20xxxxxxx", re.compile(r"\b20\d{7}\b")),
    ("original Japanese admin label", re.compile(re.escape(ADMIN_LABEL_JA))),
    ("original security-release label", re.compile(re.escape(SECURITY_RELEASE_JA))),
    ("original control prefix", re.compile(re.escape(CONTROL_PREFIX_JA))),
]


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        yield path


def scan(root: Path) -> list[tuple[str, str, str]]:
    findings: list[tuple[str, str, str]] = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="latin-1")
            except Exception:
                continue
        for label, pattern in PATTERNS:
            match = pattern.search(text)
            if match:
                findings.append((str(path.relative_to(root)), label, match.group(0)))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    findings = scan(root)
    if findings:
        print("Potential sensitive strings found:")
        for file_name, label, value in findings:
            print(f"- {file_name}: {label}: {value[:80]}")
        return 1
    print("Package check passed: no configured sensitive patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
