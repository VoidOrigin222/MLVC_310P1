#!/usr/bin/env python3
"""Fail if tooling adds cleanup paths for AOE workspaces/caches."""

from __future__ import annotations

import re
import sys
from pathlib import Path


SCAN_ROOTS = ("tools", "scripts", "CMakeLists.txt")
TEXT_SUFFIXES = {".py", ".sh", ".txt"}
CLEANUP_PATTERN = re.compile(r"\b(rmtree|unlink|remove|rmdir|removedirs)\s*\(")
AOE_PATH_PATTERN = re.compile(r"aoe_workspace|reports/aoe|reports\"?\s*/\s*\"?aoe|aoe_work_dir")

ALLOWLIST = {
    ("tools/check_aoe_workspace_policy.py", 10),
    ("tools/check_aoe_workspace_policy.py", 11),
    ("tools/check_aoe_workspace_policy.py", 12),
    ("tools/convert_acl_models.py", 295),
}


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in SCAN_ROOTS:
        path = root / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix in TEXT_SUFFIXES
            )
    return sorted(files)


def nearby_lines(lines: list[str], index: int, radius: int = 3) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(lines[start:end])


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for index, line in enumerate(lines):
            lineno = index + 1
            if (rel, lineno) in ALLOWLIST:
                continue
            if not CLEANUP_PATTERN.search(line):
                continue
            if AOE_PATH_PATTERN.search(nearby_lines(lines, index)):
                failures.append(f"{rel}:{lineno}: {line.strip()}")
    if failures:
        print("AOE workspace cleanup policy violations:", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("aoe_workspace_policy status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
