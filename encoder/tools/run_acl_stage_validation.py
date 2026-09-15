#!/usr/bin/env python3
"""Run ACL single-stage ATC-vs-AOE comparisons and write acceptance evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import check_acl_package_complete
import verify_models


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def parse_output_metrics(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {"outputs": []}
    for line in stdout.replace("\r", "\n").splitlines():
        if not line:
            continue
        if line.startswith("output="):
            output_record: dict[str, Any] = {}
            for item in line.split():
                key, value = item.split("=", 1)
                output_record[key] = coerce_value(value)
            metrics["outputs"].append(output_record)
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key] = coerce_value(value)
    return metrics


def coerce_value(value: str) -> Any:
    try:
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        return float(value)
    except ValueError:
        return value


def run_command(command: list[str], cwd: Path, log_path: Path) -> tuple[int, str, str]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n"
        + "## stdout\n"
        + completed.stdout
        + "\n## stderr\n"
        + completed.stderr,
        encoding="utf-8",
    )
    return completed.returncode, completed.stdout, completed.stderr


def resolve_binary(root: Path, build_dir: Path, binary: str) -> Path:
    path = Path(binary)
    if path.is_absolute():
        return path
    candidate = root / build_dir / path
    return candidate if candidate.exists() else root / path


def stage_names(manifest: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for index, record in enumerate(manifest.get("models", [])):
        if not isinstance(record, dict):
            raise ValueError(f"manifest models[{index}] is not an object")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"manifest models[{index}] missing name")
        names.append(name)
    return names


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--resolution", required=True, choices=("720p", "1080p"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compare-binary", default="compare_acl_stage")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=0.005)
    parser.add_argument("--stage", action="append", help="Optional stage subset.")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    source_manifest = (
        args.source_manifest if args.source_manifest.is_absolute() else root / args.source_manifest
    )
    output = args.output or root / "reports" / "acl_acceptance" / args.resolution / "single_stage_correctness.json"
    output = output if output.is_absolute() else root / output
    log_root = output.parent / "single_stage_logs"
    compare_binary = resolve_binary(root, args.build_dir, args.compare_binary)
    if not compare_binary.exists():
        raise SystemExit(f"missing compare binary: {compare_binary}")

    verify_models.verify_manifest(manifest_path, strict_assets=True, allow_missing_fused=False)
    acl_count, source_count, _ = check_acl_package_complete.check_stage_coverage(
        source_manifest,
        manifest_path,
        allow_partial=False,
    )
    manifest = load_json(manifest_path)
    requested = args.stage or stage_names(manifest)
    known = set(stage_names(manifest))
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise SystemExit("unknown --stage values: " + ", ".join(unknown))

    results: list[dict[str, Any]] = []
    for name in requested:
        log_path = log_root / f"{name}.log"
        command = [
            str(compare_binary),
            "--manifest",
            str(manifest_path),
            "--stage",
            name,
            "--device",
            str(args.device),
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
        ]
        return_code, stdout, stderr = run_command(command, root, log_path)
        metrics = parse_output_metrics(stdout)
        results.append(
            {
                "stage": name,
                "return_code": return_code,
                "status": "passed" if return_code == 0 else "failed",
                "metrics": metrics,
                "log": str(log_path.relative_to(root)),
                "stderr_tail": stderr.strip().splitlines()[-8:],
            }
        )
        if return_code != 0:
            break

    passed = all(result["return_code"] == 0 for result in results) and len(results) == len(requested)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "passed" if passed else "failed",
        "gate": "single_stage_correctness",
        "resolution": args.resolution,
        "manifest": str(manifest_path),
        "source_manifest": str(source_manifest),
        "stage_count": source_count,
        "manifest_stage_count": acl_count,
        "validated_stage_count": len(results),
        "atol": args.atol,
        "rtol": args.rtol,
        "results": results,
    }
    write_report(output, report)
    print(f"wrote {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
