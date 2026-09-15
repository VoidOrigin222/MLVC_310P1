#!/usr/bin/env python3
"""Report whether the ACL migration has enough evidence to be accepted."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import acl_conversion_status
import check_acl_package_complete
import verify_models


DEFAULT_SOURCE_ROOT = Path("/root/workspace/cgc/mlvc_cpp/models")
DEFAULT_RESOLUTIONS = ("720p", "1080p")
ACCEPTANCE_GATES = (
    "single_stage_correctness",
    "end_to_end_correctness",
    "performance",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def normalize_path(path: Path, base: Path) -> Path:
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def require_path_match(value: Any, expected: Path, base: Path, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} missing")
    actual = normalize_path(Path(value), base)
    if actual != expected.resolve(strict=False):
        raise ValueError(f"{label} must match {expected}")


def require_path_match_any_base(
    value: Any,
    expected: Path,
    bases: list[Path],
    label: str,
) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} missing")
    candidate = Path(value)
    expected_resolved = expected.resolve(strict=False)
    actuals = []
    for base in bases:
        actual = normalize_path(candidate, base)
        actuals.append(actual)
        if actual == expected_resolved:
            return
    raise ValueError(f"{label} must match {expected}")


def check_evidence(
    path: Path,
    resolution: str,
    gate: str,
    expected_stage_names: list[str] | None = None,
    expected_manifest: Path | None = None,
    expected_source_manifest: Path | None = None,
    repo_root: Path | None = None,
) -> None:
    payload = load_json(path)
    evidence_base = path.parent
    status = payload.get("status")
    if status not in {"ok", "passed"}:
        raise ValueError(f"{path} status must be ok or passed")
    actual_resolution = payload.get("resolution")
    if actual_resolution is not None and actual_resolution != resolution:
        raise ValueError(f"{path} resolution must be {resolution}")
    actual_gate = payload.get("gate")
    if actual_gate is not None and actual_gate != gate:
        raise ValueError(f"{path} gate must be {gate}")
    checks = payload.get("checks")
    if checks is not None:
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"{path} checks must be a non-empty list")
        for index, check in enumerate(checks):
            if not isinstance(check, dict) or check.get("ok") is not True:
                raise ValueError(f"{path} checks[{index}] did not pass")
    if gate == "single_stage_correctness":
        if expected_manifest is not None:
            require_path_match(payload.get("manifest"), expected_manifest, evidence_base, f"{path} manifest")
        if expected_source_manifest is not None:
            require_path_match(
                payload.get("source_manifest"),
                expected_source_manifest,
                evidence_base,
                f"{path} source_manifest",
            )
        stage_count = payload.get("stage_count")
        validated_stage_count = payload.get("validated_stage_count")
        if not isinstance(stage_count, int) or stage_count <= 0:
            raise ValueError(f"{path} missing positive stage_count")
        if expected_stage_names is not None and stage_count != len(expected_stage_names):
            raise ValueError(f"{path} stage_count must match source manifest")
        if validated_stage_count != stage_count:
            raise ValueError(f"{path} validated_stage_count must equal stage_count")
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != stage_count:
            raise ValueError(f"{path} results must contain every stage")
        stage_names: set[str] = set()
        for index, result in enumerate(results):
            if not isinstance(result, dict) or result.get("return_code") != 0:
                raise ValueError(f"{path} results[{index}] did not pass")
            stage = result.get("stage")
            if not isinstance(stage, str) or not stage:
                raise ValueError(f"{path} results[{index}] missing stage name")
            if stage in stage_names:
                raise ValueError(f"{path} duplicate stage result: {stage}")
            stage_names.add(stage)
        if expected_stage_names is not None and stage_names != set(expected_stage_names):
            raise ValueError(f"{path} stage results must match source manifest")
    elif gate == "end_to_end_correctness":
        metrics_source = payload.get("metrics_source")
        if not isinstance(metrics_source, str) or not metrics_source:
            raise ValueError(f"{path} missing metrics_source")
        metrics_path = normalize_path(Path(metrics_source), evidence_base)
        metrics = load_json(metrics_path)
        model_manifests = metrics.get("model_manifests")
        if not isinstance(model_manifests, dict):
            raise ValueError(f"{metrics_path} missing model_manifests")
        if expected_manifest is not None:
            require_path_match_any_base(
                model_manifests.get(resolution),
                expected_manifest,
                [repo_root or metrics_path.parent, metrics_path.parent],
                f"{metrics_path} model_manifests.{resolution}",
            )
        runs = payload.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"{path} missing end-to-end runs")
        profiles: set[str] = set()
        for index, run in enumerate(runs):
            if (
                not isinstance(run, dict)
                or run.get("encode_return_code") != 0
                or run.get("decode_return_code") != 0
            ):
                raise ValueError(f"{path} runs[{index}] did not pass encode/decode")
            profile = run.get("profile")
            if not isinstance(profile, str) or not profile:
                raise ValueError(f"{path} runs[{index}] missing profile")
            if profile in profiles:
                raise ValueError(f"{path} duplicate profile run: {profile}")
            profiles.add(profile)
            bitstream_bytes = run.get("bitstream_bytes")
            if not isinstance(bitstream_bytes, int) or bitstream_bytes <= 0:
                raise ValueError(f"{path} runs[{index}] missing non-empty bitstream")
            if not run.get("decoded_output_sha256"):
                raise ValueError(f"{path} runs[{index}] missing decoded output hash")
    elif gate == "performance":
        metrics_source = payload.get("metrics_source")
        if not isinstance(metrics_source, str) or not metrics_source:
            raise ValueError(f"{path} missing metrics_source")
        metrics_path = normalize_path(Path(metrics_source), evidence_base)
        metrics_payload = load_json(metrics_path)
        model_manifests = metrics_payload.get("model_manifests")
        if not isinstance(model_manifests, dict):
            raise ValueError(f"{metrics_path} missing model_manifests")
        if expected_manifest is not None:
            require_path_match_any_base(
                model_manifests.get(resolution),
                expected_manifest,
                [repo_root or metrics_path.parent, metrics_path.parent],
                f"{metrics_path} model_manifests.{resolution}",
            )
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"{path} missing performance metrics")
        for key in ("encode_fps_min", "decode_fps_min"):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{path} metric {key} must be positive")
        stage_report_value = payload.get("stage_performance_report")
        if not isinstance(stage_report_value, str) or not stage_report_value:
            raise ValueError(f"{path} missing stage_performance_report")
        stage_report = Path(stage_report_value)
        if not stage_report.is_absolute():
            stage_report = path.parent / stage_report
        stage_payload = load_json(stage_report)
        if expected_manifest is not None:
            require_path_match(
                stage_payload.get("manifest"),
                expected_manifest,
                stage_report.parent,
                f"{stage_report} manifest",
            )
        if expected_source_manifest is not None:
            require_path_match(
                stage_payload.get("source_manifest"),
                expected_source_manifest,
                stage_report.parent,
                f"{stage_report} source_manifest",
            )
        if stage_payload.get("status") not in {"ok", "passed"}:
            raise ValueError(f"{stage_report} status must be ok or passed")
        if stage_payload.get("gate") != "stage_performance":
            raise ValueError(f"{stage_report} gate must be stage_performance")
        stage_count = stage_payload.get("stage_count")
        benchmarked = stage_payload.get("benchmarked_stage_count")
        if not isinstance(stage_count, int) or stage_count <= 0 or benchmarked != stage_count:
            raise ValueError(f"{stage_report} must benchmark every stage")
        if expected_stage_names is not None and stage_count != len(expected_stage_names):
            raise ValueError(f"{stage_report} stage_count must match source manifest")
        results = stage_payload.get("results")
        if not isinstance(results, list) or len(results) != stage_count:
            raise ValueError(f"{stage_report} results must contain every stage")
        stage_names: set[str] = set()
        for index, result in enumerate(results):
            if not isinstance(result, dict) or result.get("return_code") != 0:
                raise ValueError(f"{stage_report} results[{index}] did not pass")
            stage = result.get("stage")
            if not isinstance(stage, str) or not stage:
                raise ValueError(f"{stage_report} results[{index}] missing stage name")
            if stage in stage_names:
                raise ValueError(f"{stage_report} duplicate stage result: {stage}")
            stage_names.add(stage)
            avg_ms = result.get("avg_ms")
            if not isinstance(avg_ms, (int, float)) or avg_ms <= 0:
                raise ValueError(f"{stage_report} results[{index}] missing positive avg_ms")
        if expected_stage_names is not None and stage_names != set(expected_stage_names):
            raise ValueError(f"{stage_report} stage results must match source manifest")


def criterion(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def error_text(error: BaseException) -> str:
    text = str(error)
    return text if text else error.__class__.__name__


def audit_resolution(
    resolution: str,
    source_root: Path,
    models_root: Path,
    acceptance_root: Path,
) -> dict[str, Any]:
    source_manifest = source_root / resolution / "manifest.json"
    package_root = models_root / f"{resolution}_acl"
    final_manifest = package_root / "manifest.json"
    checks: list[dict[str, Any]] = []

    checks.append(
        criterion(
            "source_manifest",
            source_manifest.is_file(),
            str(source_manifest),
        )
    )
    checks.append(
        criterion(
            "acl_package_directory",
            package_root.is_dir(),
            str(package_root),
        )
    )
    checks.append(
        criterion(
            "final_manifest",
            final_manifest.is_file(),
            str(final_manifest),
        )
    )

    conversion: dict[str, Any] | None = None
    source_stage_names: list[str] | None = None
    if source_manifest.is_file():
        try:
            source_stage_names = check_acl_package_complete.model_names(
                check_acl_package_complete.load_json(source_manifest),
                "source manifest",
            )
        except Exception:
            source_stage_names = None
        try:
            conversion = acl_conversion_status.package_status(
                resolution,
                source_manifest,
                models_root,
                now=time.time(),
                stale_seconds=900,
            )
            tuned = int(conversion["tuned_stage_count"])
            total = int(conversion["stage_count"])
            checks.append(
                criterion(
                    "all_stage_artifacts",
                    total > 0 and tuned == total and bool(conversion["manifest_final"]["exists"]),
                    f"tuned={tuned}/{total} final_manifest={conversion['manifest_final']['exists']}",
                )
            )
        except Exception as error:  # noqa: BLE001
            checks.append(criterion("all_stage_artifacts", False, error_text(error)))

    if final_manifest.is_file():
        try:
            verify_models.verify_manifest(
                final_manifest,
                strict_assets=True,
                allow_missing_fused=False,
            )
            checks.append(criterion("strict_manifest_assets", True, str(final_manifest)))
        except Exception as error:  # noqa: BLE001
            checks.append(criterion("strict_manifest_assets", False, error_text(error)))

        if source_manifest.is_file():
            try:
                acl_count, source_count, mode = check_acl_package_complete.check_stage_coverage(
                    source_manifest,
                    final_manifest,
                    allow_partial=False,
                )
                checks.append(
                    criterion(
                        "complete_stage_coverage",
                        True,
                        f"mode={mode} stages={acl_count}/{source_count}",
                    )
                )
            except Exception as error:  # noqa: BLE001
                checks.append(criterion("complete_stage_coverage", False, error_text(error)))

    for gate in ACCEPTANCE_GATES:
        evidence = acceptance_root / resolution / f"{gate}.json"
        try:
            check_evidence(
                evidence,
                resolution,
                gate,
                source_stage_names,
                final_manifest if final_manifest.is_file() else None,
                source_manifest if source_manifest.is_file() else None,
                models_root.resolve(strict=False).parent,
            )
            checks.append(criterion(f"evidence_{gate}", True, str(evidence)))
        except Exception as error:  # noqa: BLE001
            checks.append(criterion(f"evidence_{gate}", False, error_text(error)))

    return {
        "resolution": resolution,
        "source_manifest": str(source_manifest),
        "package_root": str(package_root),
        "final_manifest": str(final_manifest),
        "conversion": conversion,
        "checks": checks,
        "ready": all(check["ok"] for check in checks),
    }


def audit_goal(
    source_root: Path,
    models_root: Path,
    acceptance_root: Path,
    resolutions: tuple[str, ...],
) -> dict[str, Any]:
    resolution_reports = [
        audit_resolution(resolution, source_root, models_root, acceptance_root)
        for resolution in resolutions
    ]
    return {
        "ready": all(report["ready"] for report in resolution_reports),
        "resolutions": resolution_reports,
        "acceptance_root": str(acceptance_root),
    }


def print_human(report: dict[str, Any]) -> None:
    state = "ready" if report["ready"] else "not_ready"
    print(f"acl_goal_readiness status={state}")
    for resolution in report["resolutions"]:
        res_state = "ready" if resolution["ready"] else "not_ready"
        print(f"{resolution['resolution']}: {res_state}")
        for check in resolution["checks"]:
            mark = "ok" if check["ok"] else "missing"
            print(f"  {mark} {check['name']}: {check['detail']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--acceptance-root", type=Path, default=Path("reports/acl_acceptance"))
    parser.add_argument(
        "--resolution",
        action="append",
        choices=DEFAULT_RESOLUTIONS,
        help="Resolution to audit. Defaults to both 720p and 1080p.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full audit as JSON.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless every readiness criterion passes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolutions = tuple(args.resolution or DEFAULT_RESOLUTIONS)
    report = audit_goal(
        args.source_root,
        args.models_root,
        args.acceptance_root,
        resolutions,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report)
    return 1 if args.require_complete and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
