#!/usr/bin/env python3
"""Regression checks for ACL migration readiness reporting."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import acl_goal_readiness as readiness


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_file(path: Path, payload: bytes = b"x") -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path} did not contain a JSON object")
    return value


def make_source_manifest(source_root: Path, resolution: str) -> None:
    manifest = {
        "created_at": "2026-05-16T00:00:00Z",
        "dtype": "fp16",
        "sidecar": {"file": "sidecars.mlvcsc", "bytes": 1, "sha256": sha256_bytes(b"s")},
        "models": [
            {
                "name": "stage_a",
                "file": "stage_a.sim.onnx",
                "bytes": 1,
                "sha256": sha256_bytes(b"o"),
                "frame_type": "I",
                "route": "encode",
                "inputs": [{"name": "x", "dtype": "float16", "shape": [1, 1, 1, 1]}],
                "outputs": [{"name": "y", "dtype": "float16", "shape": [1, 1, 1, 1]}],
            }
        ],
    }
    root = source_root / resolution
    write_file(root / "sidecars.mlvcsc", b"s")
    write_file(root / "stage_a.sim.onnx", b"o")
    write_json(root / "manifest.json", manifest)


def make_acl_package(models_root: Path, resolution: str) -> None:
    root = models_root / f"{resolution}_acl"
    sidecar = write_file(root / "sidecars.mlvcsc", b"s")
    original = write_file(root / "onnx_original" / "stage_a.sim.onnx", b"o")
    write_file(root / "onnx_optimized" / "stage_a.sim.onnx", b"p")
    write_file(root / "om_atc" / "stage_a.sim.om", b"a")
    aoe = write_file(root / "om_aoe" / "stage_a.sim.om", b"e")
    write_file(root / "reports" / "io" / "stage_a.sim.atc.json", b"{}")
    write_file(root / "reports" / "io" / "stage_a.sim.aoe.json", b"{}")
    write_json(root / "reports" / "rewrite" / "stage_a.sim.json", {"match_count": 2})
    manifest = {
        "created_at": "2026-05-16T00:00:00Z",
        "runtime": "acl",
        "soc_version": "Ascend310P3",
        "dtype": "fp16",
        "sidecar": {"file": "sidecars.mlvcsc", **sidecar},
        "models": [
            {
                "name": "stage_a",
                "file": "om_aoe/stage_a.sim.om",
                "onnx_file": "onnx_original/stage_a.sim.onnx",
                "optimized_onnx_file": "onnx_optimized/stage_a.sim.onnx",
                "atc_om_file": "om_atc/stage_a.sim.om",
                "backend": "acl",
                "bytes": aoe["bytes"],
                "sha256": aoe["sha256"],
                "frame_type": "I",
                "route": "encode",
                "optimizations": {
                    "post_sigmoid_chunk_add": 2,
                    "fp16_io": True,
                    "aoe": "highest",
                    "aoe_tune_optimization_level": "O1",
                },
                "inputs": [{"name": "x", "dtype": "float16", "shape": [1, 1, 1, 1]}],
                "outputs": [{"name": "y", "dtype": "float16", "shape": [1, 1, 1, 1]}],
            }
        ],
    }
    assert original["bytes"] == 1
    write_json(root / "manifest.json", manifest)


def make_acceptance(acceptance_root: Path, resolution: str) -> None:
    metrics_path = acceptance_root / resolution / "metrics.json"
    source_root = acceptance_root.parent / "source"
    models_root = acceptance_root.parent / "models"
    write_json(
        metrics_path,
        {
            "model_manifests": {
                resolution: f"models/{resolution}_acl/manifest.json",
            },
            "runs": [],
        },
    )
    write_json(
        acceptance_root / resolution / "stage_performance.json",
        {
            "status": "passed",
            "resolution": resolution,
            "gate": "stage_performance",
            "manifest": str(models_root / f"{resolution}_acl" / "manifest.json"),
            "source_manifest": str(source_root / resolution / "manifest.json"),
            "stage_count": 1,
            "benchmarked_stage_count": 1,
            "results": [{"stage": "stage_a", "return_code": 0, "avg_ms": 0.01}],
        },
    )
    write_json(
        acceptance_root / resolution / "single_stage_correctness.json",
        {
            "status": "passed",
            "resolution": resolution,
            "gate": "single_stage_correctness",
            "manifest": str(models_root / f"{resolution}_acl" / "manifest.json"),
            "source_manifest": str(source_root / resolution / "manifest.json"),
            "stage_count": 1,
            "validated_stage_count": 1,
            "results": [{"stage": "stage_a", "return_code": 0}],
        },
    )
    write_json(
        acceptance_root / resolution / "end_to_end_correctness.json",
        {
            "status": "passed",
            "resolution": resolution,
            "gate": "end_to_end_correctness",
            "metrics_source": str(metrics_path),
            "runs": [
                {
                    "profile": "pipeline-v1",
                    "encode_return_code": 0,
                    "decode_return_code": 0,
                    "bitstream_bytes": 1,
                    "decoded_output_sha256": "decoded",
                }
            ],
        },
    )
    write_json(
        acceptance_root / resolution / "performance.json",
        {
            "status": "passed",
            "resolution": resolution,
            "gate": "performance",
            "metrics_source": str(metrics_path),
            "stage_performance_report": "stage_performance.json",
            "metrics": {
                "encode_fps_min": 1.0,
                "decode_fps_min": 1.0,
            },
        },
    )


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_not_ready(report: dict[str, Any], message: str) -> None:
    assert_true(not report["ready"], message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mlvc_acl_goal_readiness_") as temp:
        root = Path(temp)
        source_root = root / "source"
        models_root = root / "models"
        acceptance_root = root / "acceptance"
        for resolution in readiness.DEFAULT_RESOLUTIONS:
            make_source_manifest(source_root, resolution)
            make_acl_package(models_root, resolution)
            make_acceptance(acceptance_root, resolution)

        report = readiness.audit_goal(
            source_root,
            models_root,
            acceptance_root,
            readiness.DEFAULT_RESOLUTIONS,
        )
        assert_true(report["ready"], "complete fixture should be ready")

        single_stage = acceptance_root / "720p" / "single_stage_correctness.json"
        duplicate_stage_payload = load_json(single_stage)
        duplicate_stage_payload["results"].append(dict(duplicate_stage_payload["results"][0]))
        duplicate_stage_payload["stage_count"] = 2
        duplicate_stage_payload["validated_stage_count"] = 2
        write_json(single_stage, duplicate_stage_payload)
        report = readiness.audit_goal(
            source_root,
            models_root,
            acceptance_root,
            readiness.DEFAULT_RESOLUTIONS,
        )
        assert_not_ready(report, "duplicate single-stage evidence should block readiness")
        make_acceptance(acceptance_root, "720p")

        wrong_stage_payload = load_json(single_stage)
        wrong_stage_payload["results"][0]["stage"] = "stage_z"
        write_json(single_stage, wrong_stage_payload)
        report = readiness.audit_goal(
            source_root,
            models_root,
            acceptance_root,
            readiness.DEFAULT_RESOLUTIONS,
        )
        assert_not_ready(report, "wrong single-stage evidence stage should block readiness")
        make_acceptance(acceptance_root, "720p")

        stale_manifest_payload = load_json(single_stage)
        stale_manifest_payload["manifest"] = str(models_root / "stale_720p_acl" / "manifest.json")
        write_json(single_stage, stale_manifest_payload)
        report = readiness.audit_goal(
            source_root,
            models_root,
            acceptance_root,
            readiness.DEFAULT_RESOLUTIONS,
        )
        assert_not_ready(report, "stale single-stage manifest evidence should block readiness")
        make_acceptance(acceptance_root, "720p")

        stage_performance = acceptance_root / "720p" / "stage_performance.json"
        bad_stage_perf = load_json(stage_performance)
        bad_stage_perf["results"][0]["avg_ms"] = 0
        write_json(stage_performance, bad_stage_perf)
        report = readiness.audit_goal(
            source_root,
            models_root,
            acceptance_root,
            readiness.DEFAULT_RESOLUTIONS,
        )
        assert_not_ready(report, "non-positive stage benchmark timing should block readiness")
        make_acceptance(acceptance_root, "720p")

        end_to_end = acceptance_root / "720p" / "end_to_end_correctness.json"
        bad_e2e = load_json(end_to_end)
        bad_e2e["runs"][0]["bitstream_bytes"] = 0
        write_json(end_to_end, bad_e2e)
        report = readiness.audit_goal(
            source_root,
            models_root,
            acceptance_root,
            readiness.DEFAULT_RESOLUTIONS,
        )
        assert_not_ready(report, "empty end-to-end bitstream should block readiness")
        make_acceptance(acceptance_root, "720p")

        missing_evidence = acceptance_root / "1080p" / "performance.json"
        missing_evidence.unlink()
        report = readiness.audit_goal(
            source_root,
            models_root,
            acceptance_root,
            readiness.DEFAULT_RESOLUTIONS,
        )
        assert_not_ready(report, "missing acceptance evidence should block readiness")
        checks = report["resolutions"][1]["checks"]
        assert_true(
            any(check["name"] == "evidence_performance" and not check["ok"] for check in checks),
            "performance evidence failure was not reported",
        )

        (models_root / "720p_acl" / "manifest.json").unlink()
        report = readiness.audit_goal(
            source_root,
            models_root,
            acceptance_root,
            readiness.DEFAULT_RESOLUTIONS,
        )
        assert_not_ready(report, "missing final manifest should block readiness")
        checks = report["resolutions"][0]["checks"]
        assert_true(
            any(check["name"] == "final_manifest" and not check["ok"] for check in checks),
            "final manifest failure was not reported",
        )

    print("acl_goal_readiness_check status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
