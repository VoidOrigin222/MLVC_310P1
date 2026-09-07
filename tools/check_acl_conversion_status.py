#!/usr/bin/env python3
"""Regression checks for side-effect-free ACL conversion status reporting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import acl_conversion_status as status


def write_file(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def make_source_manifest(source_root: Path) -> None:
    manifest = {
        "models": [
            {"name": "complete", "file": "complete.sim.onnx"},
            {"name": "missing_atc_io", "file": "missing_atc_io.sim.onnx"},
        ]
    }
    manifest_path = source_root / "720p" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def make_stage(package_root: Path, stem: str, include_atc_io: bool) -> None:
    write_file(package_root / "onnx_original" / f"{stem}.onnx")
    write_file(package_root / "onnx_optimized" / f"{stem}.onnx")
    write_file(package_root / "reports" / "rewrite" / f"{stem}.json", b'{"match_count": 0}')
    write_file(package_root / "om_atc" / f"{stem}.om")
    if include_atc_io:
        write_file(package_root / "reports" / "io" / f"{stem}.atc.json", b"{}")
    write_file(package_root / "om_aoe" / f"{stem}.om")
    write_file(package_root / "reports" / "io" / f"{stem}.aoe.json", b"{}")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mlvc_acl_conversion_status_") as temp:
        root = Path(temp)
        source_root = root / "source"
        acl_root = root / "models"
        make_source_manifest(source_root)
        package_root = acl_root / "720p_acl"
        make_stage(package_root, "complete.sim", include_atc_io=True)
        make_stage(package_root, "missing_atc_io.sim", include_atc_io=False)

        report = status.package_status(
            "720p",
            source_root / "720p" / "manifest.json",
            acl_root,
            now=0.0,
            stale_seconds=900,
        )
        assert_true(report["stage_count"] == 2, "unexpected stage count")
        assert_true(report["tuned_stage_count"] == 1, "status should require full artifact set")
        assert_true(report["tuned_stages"] == ["complete"], "unexpected tuned stages")
        assert_true(report["remaining_stages"] == ["missing_atc_io"], "unexpected remaining stages")
        assert_true(
            report["missing_by_stage"] == {"missing_atc_io": ["atc_io_json"]},
            "missing stage should report only the absent ATC I/O JSON",
        )

    print("acl_conversion_status_check status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
