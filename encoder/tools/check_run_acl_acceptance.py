#!/usr/bin/env python3
"""Regression checks for ACL acceptance runner argument behavior."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mlvc_acl_acceptance_runner_") as temp:
        root = Path(temp)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_acl_acceptance.py"),
                "--root",
                str(root),
                "--resolution",
                "720p",
                "--skip-end-to-end",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert_true(completed.returncode != 0, "runner should fail without final manifest")
        assert_true("missing final ACL manifest" in completed.stderr + completed.stdout,
                    "missing-manifest error was not reported")

        source_manifest = root / "source" / "720p" / "manifest.json"
        final_manifest = root / "models" / "720p_acl" / "manifest.json"
        source_manifest.parent.mkdir(parents=True, exist_ok=True)
        final_manifest.parent.mkdir(parents=True, exist_ok=True)
        source_manifest.write_text('{"models": [{"name": "stage_a"}]}', encoding="utf-8")
        final_manifest.write_text('{"models": [{"name": "stage_a"}]}', encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_acl_acceptance.py"),
                "--root",
                str(root),
                "--source-root",
                str(root / "source"),
                "--models-root",
                str(root / "models"),
                "--acceptance-root",
                str(root / "acceptance"),
                "--build-dir",
                "custom_build",
                "--resolution",
                "720p",
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        assert_true(completed.returncode == 0, output)
        assert_true("--build-dir custom_build" in output, "stage tools did not receive build-dir")
        assert_true("--manifest-720p" in output and str(final_manifest) in output,
                    "baseline did not receive selected manifest")
        assert_true("--acceptance-output-root" in output and str(root / "acceptance") in output,
                    "baseline did not receive acceptance root")
        assert_true("--models-root" in output and str(root / "models") in output,
                    "readiness command did not receive models root")
        assert_true("--acceptance-root" in output and str(root / "acceptance") in output,
                    "readiness command did not receive acceptance root")

    print("run_acl_acceptance_check status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
