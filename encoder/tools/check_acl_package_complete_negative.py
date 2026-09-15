#!/usr/bin/env python3
"""Regression checks for ACL package stage-coverage policy."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import check_acl_package_complete as coverage


def write_manifest(path: Path, names: list[str]) -> Path:
    manifest: dict[str, Any] = {
        "models": [{"name": name, "file": f"{name}.sim.onnx"} for name in names]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def expect_failure(source: Path, acl: Path, allow_partial: bool, expected: str) -> None:
    try:
        coverage.check_stage_coverage(source, acl, allow_partial)
    except Exception as error:  # noqa: BLE001
        if expected not in str(error):
            raise AssertionError(f"expected {expected!r} in {error!s}") from error
        return
    raise AssertionError(f"stage coverage unexpectedly passed: {acl}")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mlvc_acl_package_complete_") as temp:
        root = Path(temp)
        source = write_manifest(root / "source.json", ["a", "b", "c"])
        complete = write_manifest(root / "complete.json", ["a", "b", "c"])
        partial = write_manifest(root / "partial.json", ["a", "c"])
        out_of_order = write_manifest(root / "out_of_order.json", ["b", "a"])
        unknown = write_manifest(root / "unknown.json", ["a", "z"])
        duplicate = write_manifest(root / "duplicate.json", ["a", "a"])

        assert_true(
            coverage.check_stage_coverage(source, complete, allow_partial=False) == (3, 3, "complete"),
            "complete manifest should pass",
        )
        assert_true(
            coverage.check_stage_coverage(source, partial, allow_partial=True) == (2, 3, "partial"),
            "ordered partial manifest should pass in partial mode",
        )
        expect_failure(source, partial, allow_partial=False, expected="ACL manifest is not complete")
        expect_failure(source, out_of_order, allow_partial=True, expected="stage order")
        expect_failure(source, unknown, allow_partial=True, expected="stages not present")
        expect_failure(source, duplicate, allow_partial=True, expected="duplicated model name")

    print("acl_package_complete_negative status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
