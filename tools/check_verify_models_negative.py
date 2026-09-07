#!/usr/bin/env python3
"""Negative tests for the ACL-aware model manifest verifier."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path

import verify_models


def sha256_and_size(path: Path) -> tuple[int, str]:
    return verify_models.sha256_file(path)


def write_file(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    size, digest = sha256_and_size(path)
    return {"bytes": size, "sha256": digest}


def base_manifest(root: Path) -> dict[str, object]:
    sidecar_meta = write_file(root / "sidecars.mlvcsc", b"sidecar")
    om_meta = write_file(root / "om_aoe" / "stage.sim.om", b"aoe")
    write_file(root / "onnx_original" / "stage.sim.onnx", b"onnx")
    write_file(root / "onnx_optimized" / "stage.sim.onnx", b"optimized")
    write_file(root / "om_atc" / "stage.sim.om", b"atc")
    write_file(root / "reports" / "io" / "stage.sim.atc.json", b"{}")
    write_file(root / "reports" / "io" / "stage.sim.aoe.json", b"{}")
    write_file(root / "reports" / "rewrite" / "stage.sim.json", b'{"match_count": 0}')
    return {
        "created_at": "2026-05-16T00:00:00+00:00",
        "runtime": "acl",
        "soc_version": "Ascend310P3",
        "dtype": "fp16",
        "sidecar": {
            "file": "sidecars.mlvcsc",
            "bytes": sidecar_meta["bytes"],
            "sha256": sidecar_meta["sha256"],
        },
        "models": [
            {
                "name": "stage",
                "file": "om_aoe/stage.sim.om",
                "onnx_file": "onnx_original/stage.sim.onnx",
                "optimized_onnx_file": "onnx_optimized/stage.sim.onnx",
                "atc_om_file": "om_atc/stage.sim.om",
                "backend": "acl",
                "bytes": om_meta["bytes"],
                "sha256": om_meta["sha256"],
                "frame_type": "I",
                "route": "encode",
                "inputs": [{"name": "x", "dtype": "float16", "shape": [1, 1, 1, 1]}],
                "outputs": [{"name": "y", "dtype": "float16", "shape": [1, 1, 1, 1]}],
                "optimizations": {
                    "post_sigmoid_chunk_add": 0,
                    "fp16_io": True,
                    "aoe": "highest",
                    "aoe_tune_optimization_level": "O1",
                },
            }
        ],
    }


def write_manifest(root: Path, manifest: dict[str, object], name: str) -> Path:
    path = root / name
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def expect_failure(path: Path, expected: str) -> None:
    try:
        verify_models.verify_manifest(path, strict_assets=False, allow_missing_fused=False)
    except Exception as error:  # noqa: BLE001
        if expected not in str(error):
            raise AssertionError(f"expected {expected!r} in {error!s}") from error
        return
    raise AssertionError(f"manifest unexpectedly passed: {path}")


def main() -> int:
    temp_root = Path(tempfile.mkdtemp(prefix="mlvc_verify_models_negative_"))
    try:
        manifest = base_manifest(temp_root)
        valid_path = write_manifest(temp_root, manifest, "valid.json")
        verify_models.verify_manifest(valid_path, strict_assets=False, allow_missing_fused=False)

        checked_cases = 0
        cases: list[tuple[str, dict[str, object], str]] = []
        bad_file = copy.deepcopy(manifest)
        bad_file["models"][0]["file"] = "om_atc/stage.sim.om"  # type: ignore[index]
        cases.append(("bad_file.json", bad_file, "file must point to om_aoe/*.om"))

        bad_backend = copy.deepcopy(manifest)
        bad_backend["models"][0]["backend"] = "cpu"  # type: ignore[index]
        cases.append(("bad_backend.json", bad_backend, "backend = acl"))

        missing_aoe = copy.deepcopy(manifest)
        del missing_aoe["models"][0]["optimizations"]["aoe_tune_optimization_level"]  # type: ignore[index]
        cases.append(("missing_aoe_level.json", missing_aoe, "aoe_tune_optimization_level"))

        bad_dtype = copy.deepcopy(manifest)
        bad_dtype["models"][0]["inputs"][0]["dtype"] = "float32"  # type: ignore[index]
        cases.append(("bad_dtype.json", bad_dtype, "must use float16 ACL I/O"))

        bad_manifest_dtype = copy.deepcopy(manifest)
        bad_manifest_dtype["dtype"] = "fp32"
        cases.append(("bad_manifest_dtype.json", bad_manifest_dtype, "ACL manifest dtype must be fp16"))

        bad_rewrite_count = copy.deepcopy(manifest)
        bad_rewrite_count["models"][0]["optimizations"]["post_sigmoid_chunk_add"] = 9  # type: ignore[index]
        cases.append(("bad_rewrite_count.json", bad_rewrite_count, "rewrite count mismatch"))

        missing_io_root = temp_root / "missing_io"
        missing_io_manifest = base_manifest(missing_io_root)
        (missing_io_root / "reports" / "io" / "stage.sim.aoe.json").unlink()
        expect_failure(
            write_manifest(missing_io_root, missing_io_manifest, "missing_aoe_io.json"),
            "AOE I/O JSON missing file",
        )
        checked_cases += 1

        extra_report_root = temp_root / "extra_report"
        extra_report_manifest = base_manifest(extra_report_root)
        write_file(extra_report_root / "reports" / "io" / "unlisted.sim.aoe.json", b"{}")
        extra_report_path = write_manifest(extra_report_root, extra_report_manifest, "extra_report.json")
        try:
            verify_models.verify_manifest(extra_report_path, strict_assets=True, allow_missing_fused=False)
        except Exception as error:  # noqa: BLE001
            if "manifest does not list assets" not in str(error):
                raise AssertionError(f"unexpected extra-report error: {error!s}") from error
        else:
            raise AssertionError("strict verifier accepted an unlisted report asset")
        checked_cases += 1

        for filename, bad_manifest, expected in cases:
            expect_failure(write_manifest(temp_root, bad_manifest, filename), expected)
            checked_cases += 1

        print(f"verify_models_negative status=ok cases={checked_cases}")
        return 0
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
