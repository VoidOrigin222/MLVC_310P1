#!/usr/bin/env python3
"""Check performance baseline manifest selection without running encode/decode."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import run_performance_baseline as perf


def namespace(**kwargs: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "manifest_720p": None,
        "manifest_1080p": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def expect_missing_default(root: Path, resolution: str) -> None:
    try:
        perf.selected_manifest(namespace(), resolution, root)
    except SystemExit as error:
        message = str(error)
        expected = f"missing ACL manifest for {resolution}:"
        if expected not in message or f"--manifest-{resolution}" not in message:
            raise AssertionError(f"unexpected missing-manifest message: {message}")
        return
    raise AssertionError(f"default {resolution} manifest unexpectedly existed under {root}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mlvc_perf_manifest_policy_") as temp:
        root = Path(temp)
        expect_missing_default(root, "720p")
        expect_missing_default(root, "1080p")

        override = root / "models" / "720p_acl" / "manifest.partial.json"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text("{}", encoding="utf-8")
        selected = perf.selected_manifest(
            namespace(manifest_720p=str(override.relative_to(root))), "720p", root
        )
        if selected != "models/720p_acl/manifest.partial.json":
            raise AssertionError(f"unexpected selected manifest: {selected}")

        passing_results = {
            "runs": [
                {
                    "encode": {"return_code": 0},
                    "decode": {"return_code": 0},
                }
            ],
            "correctness": [
                {
                    "bitstream_matches_reference": True,
                    "decoded_png_matches_reference": True,
                }
            ],
        }
        failing_results = {
            "runs": [
                {
                    "encode": {"return_code": 1},
                    "decode": {"return_code": 0},
                }
            ],
            "correctness": [
                {
                    "bitstream_matches_reference": True,
                    "decoded_png_matches_reference": True,
                }
            ],
        }
        if not perf.baseline_passed(passing_results):
            raise AssertionError("passing baseline results were rejected")
        if perf.baseline_passed(failing_results):
            raise AssertionError("failing baseline results were accepted")

    print("performance_baseline_manifest_policy status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
