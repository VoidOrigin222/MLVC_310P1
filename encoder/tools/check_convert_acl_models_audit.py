#!/usr/bin/env python3
"""Regression checks for convert_acl_models audit-only bookkeeping."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import convert_acl_models as convert


def write_file(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def dirs_for(root: Path) -> dict[str, Path]:
    return {
        "onnx_original": root / "onnx_original",
        "onnx_optimized": root / "onnx_optimized",
        "om_atc": root / "om_atc",
        "om_aoe": root / "om_aoe",
        "reports": root / "reports",
        "rewrite": root / "reports" / "rewrite",
        "atc": root / "reports" / "atc",
        "aoe": root / "reports" / "aoe",
        "io": root / "reports" / "io",
    }


def make_source_manifest(source_root: Path, manifest_path: Path) -> dict[str, Any]:
    write_file(source_root / "sidecars.mlvcsc", b"sidecar")
    for stage in ("stage_a", "stage_b"):
        write_file(source_root / f"{stage}.sim.onnx", stage.encode("ascii"))
    manifest = {
        "dtype": "fp16",
        "sidecar": {"file": "sidecars.mlvcsc"},
        "models": [
            {"name": "stage_a", "file": "stage_a.sim.onnx", "frame_type": "I", "route": "encode"},
            {"name": "stage_b", "file": "stage_b.sim.onnx", "frame_type": "P", "route": "decode"},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def make_complete_stage(output_root: Path, stage: str, match_count: int) -> None:
    dirs = dirs_for(output_root)
    stem = f"{stage}.sim"
    write_file(dirs["onnx_original"] / f"{stem}.onnx", b"original")
    write_file(dirs["onnx_optimized"] / f"{stem}.onnx", b"optimized")
    write_file(dirs["om_atc"] / f"{stem}.om", b"atc")
    write_file(dirs["om_aoe"] / f"{stem}.om", b"aoe")
    write_file(dirs["io"] / f"{stem}.atc.json", b"{}")
    write_file(dirs["io"] / f"{stem}.aoe.json", b"{}")
    rewrite_report = {"op_type": "PostSigmoidChunkAdd", "match_count": match_count}
    write_file(dirs["rewrite"] / f"{stem}.json", json.dumps(rewrite_report).encode("utf-8"))
    write_file(dirs["aoe"] / stem / "aoe_workspace" / "cache.bin", b"cache")


def make_stage_without_atc_io(output_root: Path, stage: str, match_count: int) -> None:
    make_complete_stage(output_root, stage, match_count)
    os.remove(dirs_for(output_root)["io"] / f"{stage}.sim.atc.json")


def fake_tensor_specs(_path: Path, kind: str) -> list[dict[str, Any]]:
    if kind == "input":
        return [{"name": "x", "dtype": "float16", "shape": [1, 1, 2, 2]}]
    return [{"name": "z_symbols", "dtype": "int8", "shape": [1, 1, 2, 2]}]


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mlvc_convert_audit_") as temp:
        temp_root = Path(temp)
        source_root = temp_root / "source"
        manifest_path = source_root / "manifest.json"
        source_root.mkdir(parents=True)
        manifest = make_source_manifest(source_root, manifest_path)
        args = argparse.Namespace(manifest=manifest_path, soc="Ascend310P3", aoe_mode="highest")

        original_tensor_specs = convert.tensor_specs
        original_capture_command = convert.capture_command
        convert.tensor_specs = fake_tensor_specs
        convert.capture_command = lambda command: " ".join(command) + "\n"
        try:
            metadata = convert.tool_metadata("/fake/atc", "/fake/aoe")
            assert_true(metadata["atc_version"] == "/fake/atc --version\n", "missing ATC version capture")
            assert_true(metadata["aoe_version"] == "/fake/aoe --version\n", "missing AOE version capture")
            assert_true(metadata["atc_help"] == "/fake/atc --help\n", "missing ATC help capture")
            assert_true(metadata["aoe_help"] == "/fake/aoe --help\n", "missing AOE help capture")
            convert.validate_args(argparse.Namespace(skip_atc=False, skip_aoe=False, audit_only=False))
            convert.validate_args(argparse.Namespace(skip_atc=True, skip_aoe=True, audit_only=True))
            try:
                convert.validate_args(argparse.Namespace(skip_atc=True, skip_aoe=False, audit_only=False))
            except RuntimeError as error:
                assert_true("ATC is required" in str(error), "unexpected --skip-atc error")
            else:
                raise AssertionError("--skip-atc was accepted for final conversion")
            try:
                convert.validate_args(argparse.Namespace(skip_atc=False, skip_aoe=True, audit_only=False))
            except RuntimeError as error:
                assert_true("AOE is required" in str(error), "unexpected --skip-aoe error")
            else:
                raise AssertionError("--skip-aoe was accepted for final conversion")

            output_root = temp_root / "acl"
            make_complete_stage(output_root, "stage_a", match_count=3)
            audit = convert.write_package_audit(
                manifest, source_root, output_root, dirs_for(output_root), set(), args
            )
            summary = audit["summary"]
            assert_true(summary["complete_stage_count"] == 1, "expected one complete stage")
            assert_true(summary["incomplete_stage_count"] == 1, "expected one incomplete stage")
            assert_true(summary["complete_stages"] == ["stage_a"], "unexpected complete stage list")
            assert_true(summary["incomplete_stages"] == ["stage_b"], "unexpected incomplete stage list")
            assert_true("stage_b" in summary["missing_by_stage"], "missing stage_b audit entry")
            assert_true(
                audit["aoe_workspace_retention"]["policy"] == "preserve",
                "audit must record AOE workspace preservation",
            )
            stage_a = next(model for model in audit["models"] if model["name"] == "stage_a")
            assert_true(
                stage_a["artifacts"]["aoe_work_dir"]["exists"],
                "audit must report existing AOE work directory",
            )

            partial = convert.rebuild_partial_manifest_from_artifacts(
                manifest, source_root, output_root, dirs_for(output_root), set(), args
            )
            partial_path = output_root / "manifest.partial.json"
            cache_file = output_root / "reports" / "aoe" / "stage_a.sim" / "aoe_workspace" / "cache.bin"
            assert_true(partial_path.is_file(), "partial manifest was not written")
            assert_true(cache_file.is_file(), "AOE cache was removed unexpectedly")
            assert_true([model["name"] for model in partial["models"]] == ["stage_a"], "bad partial stages")
            record = partial["models"][0]
            assert_true(record["file"] == "om_aoe/stage_a.sim.om", "partial must point to AOE OM")
            assert_true(record["backend"] == "acl", "partial record must use ACL backend")
            assert_true(record["optimizations"]["post_sigmoid_chunk_add"] == 3, "bad rewrite count")
            assert_true(record["optimizations"]["aoe_tune_optimization_level"] == "O1", "bad AOE level")
            assert_true(
                record["outputs"][0]["dtype"] == "float16",
                "partial runtime outputs must reflect --output_type=FP16 OM outputs",
            )

            missing_atc_io_root = temp_root / "missing_atc_io_acl"
            make_stage_without_atc_io(missing_atc_io_root, "stage_a", match_count=3)
            missing_atc_io_partial = convert.rebuild_partial_manifest_from_artifacts(
                manifest, source_root, missing_atc_io_root, dirs_for(missing_atc_io_root), set(), args
            )
            assert_true(
                missing_atc_io_partial["models"] == [],
                "rebuild must require ATC I/O metadata before admitting a stage",
            )
            assert_true(
                not (missing_atc_io_root / "manifest.partial.json").exists(),
                "missing ATC I/O metadata should not write a partial manifest",
            )

            empty_root = temp_root / "empty_acl"
            stale_partial = empty_root / "manifest.partial.json"
            stale_cache = empty_root / "reports" / "aoe" / "stage_a.sim" / "aoe_workspace" / "cache.bin"
            write_file(stale_partial, b'{"models": [{"name": "stale"}]}')
            write_file(stale_cache, b"cache")
            empty_partial = convert.rebuild_partial_manifest_from_artifacts(
                manifest, source_root, empty_root, dirs_for(empty_root), set(), args
            )
            assert_true(empty_partial["models"] == [], "empty rebuild should not keep stale models")
            assert_true(not stale_partial.exists(), "stale partial manifest should be removed")
            assert_true(stale_cache.is_file(), "empty rebuild must preserve AOE cache files")
        finally:
            convert.tensor_specs = original_tensor_specs
            convert.capture_command = original_capture_command

    print("convert_acl_models_audit status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
