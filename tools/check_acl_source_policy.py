#!/usr/bin/env python3
"""Fail if removed CUDA/ORT/NVTX execution-path terms reappear in ACL sources."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path


FORBIDDEN_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"CUDA",
        r"cuda",
        r"CUDAToolkit",
        r"OnnxRuntime",
        r"onnxruntime",
        r"\bOrt[A-Za-z_]*\b",
        r"NVTX",
        r"Nvtx",
        r"nvtx",
        r"\bGPU\b",
        r"\bgpu\b",
    )
)

SCAN_ROOTS = ("CMakeLists.txt", "cmake", "include", "src", "tools", "scripts")
TEXT_SUFFIXES = {".cc", ".cmake", ".cpp", ".h", ".hpp", ".py", ".sh", ".txt"}


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


def collect_failures(root: Path) -> list[str]:
    self_path = Path(__file__).resolve()
    failures: list[str] = []
    for path in iter_files(root):
        if path.resolve() == self_path:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(root)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    failures.append(f"{rel}:{lineno}: {line.strip()}")
                    break
    return failures


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="mlvc_acl_source_policy_") as temp:
        root = Path(temp)
        tools = root / "tools"
        tools.mkdir(parents=True)
        clean = tools / "clean.py"
        clean.write_text('print("acl")\nprint("device")\n', encoding="utf-8")
        if collect_failures(root):
            raise AssertionError("clean synthetic source unexpectedly failed")

        bad = tools / "bad.py"
        forbidden_term = "CU" + "DA"
        bad.write_text(f'print("acl")\nprint("{forbidden_term}")\n', encoding="utf-8")
        failures = collect_failures(root)
        if len(failures) != 1 or "bad.py:2" not in failures[0]:
            raise AssertionError(f"forbidden synthetic source was not detected: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        print("acl_source_policy_self_test status=ok")
        return 0

    root = Path(__file__).resolve().parents[1]
    failures = collect_failures(root)

    if failures:
        print("Forbidden legacy runtime/build terms found:", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("acl_source_policy status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
