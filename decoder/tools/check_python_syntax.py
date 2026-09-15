#!/usr/bin/env python3
"""Parse Python tools without writing __pycache__ files."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    failures: list[str] = []
    for raw_path in sys.argv[1:]:
        path = Path(raw_path)
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            failures.append(f"{path}:{error.lineno}:{error.offset}: {error.msg}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(f"python_syntax status=ok files={len(sys.argv) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
