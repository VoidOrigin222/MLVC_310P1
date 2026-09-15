#!/usr/bin/env python3
"""Regression checks for ACL acceptance-evidence export."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mlvc_acl_acceptance_") as temp:
        root = Path(temp)
        metrics = root / "metrics.json"
        output_root = root / "acceptance"
        write_json(
            metrics,
            {
                "model_manifests": {
                    "720p": "models/720p_acl/manifest.json",
                },
                "runs": [
                    {
                        "resolution": "720p",
                        "profile": "pipeline-v1",
                        "encode": {
                            "return_code": 0,
                            "bitstream_bytes": 123,
                            "bitstream_sha256": "abc",
                            "log": "encode.log",
                            "stdout_metrics": {"encode_fps": 2.5},
                        },
                        "decode": {
                            "return_code": 0,
                            "output_sha256": "def",
                            "log": "decode.log",
                            "stdout_metrics": {"decode_fps": 3.5},
                        },
                    }
                ],
                "correctness": [
                    {
                        "resolution": "720p",
                        "profile": "pipeline-v1",
                        "bitstream_matches_reference": True,
                        "decoded_png_matches_reference": True,
                    }
                ],
            },
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "export_acl_acceptance.py"),
                "--metrics",
                str(metrics),
                "--output-root",
                str(output_root),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert_true(completed.returncode == 0, completed.stderr or completed.stdout)
        end_to_end = json.loads(
            (output_root / "720p" / "end_to_end_correctness.json").read_text(encoding="utf-8")
        )
        performance = json.loads(
            (output_root / "720p" / "performance.json").read_text(encoding="utf-8")
        )
        assert_true(end_to_end["status"] == "passed", "end-to-end report should pass")
        assert_true(performance["status"] == "passed", "performance report should pass")
        assert_true(
            end_to_end["metrics_source"] == str(metrics.resolve(strict=False)),
            "end-to-end metrics_source should be absolute",
        )
        assert_true(
            performance["metrics_source"] == str(metrics.resolve(strict=False)),
            "performance metrics_source should be absolute",
        )
        assert_true(
            performance["stage_performance_report"] == "stage_performance.json",
            "missing stage-performance link",
        )
        assert_true(performance["metrics"]["encode_fps_min"] == 2.5, "bad encode FPS")
        assert_true(performance["metrics"]["decode_fps_min"] == 3.5, "bad decode FPS")

    print("export_acl_acceptance_check status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
