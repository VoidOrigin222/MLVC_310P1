#!/usr/bin/env python3
"""Verify model package files against manifest metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def require_string(record: dict[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} missing string field: {key}")
    return value


def require_int(record: dict[str, Any], key: str, label: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} missing non-negative integer field: {key}")
    return value


def optional_string_list(record: dict[str, Any], key: str, label: str) -> list[str]:
    value = record.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} field {key} must be a list of non-empty strings")
    return value


def verify_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} missing file: {path}")
    actual_bytes, actual_sha256 = sha256_file(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{label} byte mismatch: {path} expected {expected_bytes}, got {actual_bytes}"
        )
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} sha256 mismatch: {path} expected {expected_sha256}, got {actual_sha256}"
        )


def verify_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} missing file: {path}")


def resolve_asset(model_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else model_dir / path


def io_report_for(model_dir: Path, value: str, suffix: str) -> Path:
    return model_dir / "reports" / "io" / f"{Path(value).stem}.{suffix}.json"


def rewrite_report_for(model_dir: Path, value: str) -> Path:
    return model_dir / "reports" / "rewrite" / f"{Path(value).stem}.json"


def read_rewrite_match_count(path: Path, label: str) -> int:
    verify_existing_file(path, label)
    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    match_count = report.get("match_count")
    if not isinstance(match_count, int) or match_count < 0:
        raise ValueError(f"{label} missing non-negative integer match_count")
    return match_count


def require_tensor_specs(record: dict[str, Any], key: str, label: str, require_acl_io: bool) -> None:
    specs = record.get(key)
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"{label} missing non-empty tensor spec list: {key}")
    for index, spec in enumerate(specs):
        spec_label = f"{label} {key}[{index}]"
        if not isinstance(spec, dict):
            raise ValueError(f"{spec_label} is not an object")
        require_string(spec, "name", spec_label)
        dtype = require_string(spec, "dtype", spec_label)
        if require_acl_io and dtype != "float16":
            raise ValueError(f"{spec_label} must use float16 ACL I/O, got {dtype}")
        shape = spec.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or not all(isinstance(dim, int) and dim > 0 for dim in shape)
        ):
            raise ValueError(f"{spec_label} missing positive integer shape")


def verify_acl_model_record(
    model: dict[str, Any],
    name: str,
    model_dir: Path,
    expected_assets: set[Path],
) -> None:
    if model.get("backend") != "acl":
        raise ValueError(f"ACL model {name} must declare backend = acl")
    model_file = require_string(model, "file", f"ACL model {name}")
    if not model_file.startswith("om_aoe/") or not model_file.endswith(".om"):
        raise ValueError(f"ACL model {name} file must point to om_aoe/*.om: {model_file}")
    traceability_files: dict[str, str] = {}
    for key, expected_prefix, expected_suffix in (
        ("onnx_file", "onnx_original/", ".onnx"),
        ("optimized_onnx_file", "onnx_optimized/", ".onnx"),
        ("atc_om_file", "om_atc/", ".om"),
    ):
        value = require_string(model, key, f"ACL model {name}")
        if not value.startswith(expected_prefix) or not value.endswith(expected_suffix):
            raise ValueError(
                f"ACL model {name} field {key} must point to {expected_prefix}*{expected_suffix}: {value}"
            )
        path = resolve_asset(model_dir, value)
        verify_existing_file(path, f"ACL model {name} {key}")
        expected_assets.add(path.resolve())
        traceability_files[key] = value
    atc_io_report = io_report_for(model_dir, traceability_files["atc_om_file"], "atc")
    aoe_io_report = io_report_for(model_dir, model_file, "aoe")
    verify_existing_file(atc_io_report, f"ACL model {name} ATC I/O JSON")
    verify_existing_file(aoe_io_report, f"ACL model {name} AOE I/O JSON")
    expected_assets.add(atc_io_report.resolve())
    expected_assets.add(aoe_io_report.resolve())

    optimizations = model.get("optimizations")
    if not isinstance(optimizations, dict):
        raise ValueError(f"ACL model {name} missing optimizations object")
    if optimizations.get("fp16_io") is not True:
        raise ValueError(f"ACL model {name} must declare optimizations.fp16_io = true")
    if optimizations.get("aoe") != "highest":
        raise ValueError(f"ACL model {name} must declare optimizations.aoe = highest")
    rewrite_count = optimizations.get("post_sigmoid_chunk_add")
    if not isinstance(rewrite_count, int) or rewrite_count < 0:
        raise ValueError(
            f"ACL model {name} must declare non-negative "
            "optimizations.post_sigmoid_chunk_add"
        )
    rewrite_report = rewrite_report_for(model_dir, traceability_files["optimized_onnx_file"])
    actual_rewrite_count = read_rewrite_match_count(rewrite_report, f"ACL model {name} rewrite report")
    expected_assets.add(rewrite_report.resolve())
    if actual_rewrite_count != rewrite_count:
        raise ValueError(
            f"ACL model {name} rewrite count mismatch: manifest {rewrite_count}, "
            f"report {actual_rewrite_count}"
        )
    aoe_level = optimizations.get("aoe_tune_optimization_level")
    if not isinstance(aoe_level, str) or not aoe_level:
        raise ValueError(
            f"ACL model {name} must declare optimizations.aoe_tune_optimization_level"
        )
    require_tensor_specs(model, "inputs", f"ACL model {name}", require_acl_io=True)
    require_tensor_specs(model, "outputs", f"ACL model {name}", require_acl_io=True)


def verify_manifest(manifest_path: Path, strict_assets: bool, allow_missing_fused: bool) -> None:
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    model_dir = manifest_path.parent
    require_string(manifest, "created_at", str(manifest_path))
    manifest_dtype = require_string(manifest, "dtype", str(manifest_path))
    runtime = manifest.get("runtime", "")
    if runtime not in {"", "acl"}:
        raise ValueError(f"{manifest_path} has unsupported runtime: {runtime}")
    if runtime == "acl":
        require_string(manifest, "soc_version", str(manifest_path))
        if manifest_dtype not in {"fp16", "float16"}:
            raise ValueError(f"{manifest_path} ACL manifest dtype must be fp16, got {manifest_dtype}")

    sidecar = manifest.get("sidecar")
    if not isinstance(sidecar, dict):
        raise ValueError(f"{manifest_path} missing sidecar object")
    sidecar_file = require_string(sidecar, "file", "sidecar")
    sidecar_path = Path(sidecar_file)
    if not sidecar_path.is_absolute():
        sidecar_path = model_dir / sidecar_path
    verify_file(
        sidecar_path,
        require_int(sidecar, "bytes", "sidecar"),
        require_string(sidecar, "sha256", "sidecar"),
        "sidecar",
    )

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{manifest_path} missing non-empty models array")

    expected_assets = {sidecar_path.resolve()}
    seen_names: set[str] = set()
    model_records: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"{manifest_path} models[{index}] is not an object")
        name = require_string(model, "name", f"models[{index}]")
        if name in seen_names:
            raise ValueError(f"{manifest_path} duplicated model name: {name}")
        seen_names.add(name)
        model_records.append(model)

        require_string(model, "frame_type", f"model {name}")
        require_string(model, "route", f"model {name}")
        replaces = optional_string_list(model, "replaces", f"model {name}")
        model_file = model.get("file", f"{name}.sim.onnx")
        if not isinstance(model_file, str) or not model_file:
            raise ValueError(f"model {name} missing string field: file")
        if runtime == "acl":
            verify_acl_model_record(model, name, model_dir, expected_assets)
        model_path = resolve_asset(model_dir, model_file)
        if replaces and allow_missing_fused and not model_path.is_file():
            continue
        verify_file(
            model_path,
            require_int(model, "bytes", f"model {name}"),
            require_string(model, "sha256", f"model {name}"),
            f"model {name}",
        )
        expected_assets.add(model_path.resolve())

    for model in model_records:
        name = require_string(model, "name", "model")
        for replaced in optional_string_list(model, "replaces", f"model {name}"):
            if replaced not in seen_names:
                raise ValueError(f"model {name} replaces unknown stage: {replaced}")

    if strict_assets:
        if runtime == "acl":
            patterns = (
                "*.mlvcsc",
                "onnx_original/*.onnx",
                "onnx_optimized/*.onnx",
                "om_atc/*.om",
                "om_aoe/*.om",
                "reports/io/*.json",
                "reports/rewrite/*.json",
            )
        else:
            patterns = ("*.onnx", "*.mlvcsc")
        actual_assets = {
            path.resolve() for pattern in patterns for path in model_dir.glob(pattern) if path.is_file()
        }
        extra_assets = sorted(actual_assets - expected_assets)
        missing_assets = sorted(expected_assets - actual_assets)
        if extra_assets:
            raise ValueError("manifest does not list assets: " + ", ".join(map(str, extra_assets)))
        if missing_assets:
            raise ValueError(
                "manifest lists assets outside model dir: " + ", ".join(map(str, missing_assets))
            )


def discover_manifests(models_root: Path) -> list[Path]:
    return sorted(path for path in models_root.glob("*/manifest.json") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify MLVC model manifests and assets.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Manifest files or model directories. Defaults to models/*/manifest.json.",
    )
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--no-strict-assets",
        action="store_true",
        help="Do not fail on extra .onnx/.mlvcsc files in each model directory.",
    )
    parser.add_argument(
        "--allow-missing-fused",
        action="store_true",
        help="Allow fused manifest records with replaces=[...] to reference missing .onnx files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths or discover_manifests(args.models_root)
    if not paths:
        print(f"no manifests found under {args.models_root}", file=sys.stderr)
        return 2

    failed = False
    for path in paths:
        manifest_path = path / "manifest.json" if path.is_dir() else path
        try:
            verify_manifest(
                manifest_path,
                strict_assets=not args.no_strict_assets,
                allow_missing_fused=args.allow_missing_fused,
            )
            print(f"OK   {manifest_path}")
        except Exception as error:
            failed = True
            print(f"FAIL {manifest_path}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
