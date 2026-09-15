#!/usr/bin/env python3
"""Run ACL single-stage benchmarks and write stage-performance acceptance evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
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


def coerce_value(value: str) -> Any:
    try:
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_key_values(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in stdout.replace("\r", "\n").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key] = coerce_value(value)
    return metrics


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
    parser.add_argument("--benchmark-binary", default="benchmark_acl_stage")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--stage", action="append", help="Optional stage subset.")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    source_manifest = (
        args.source_manifest if args.source_manifest.is_absolute() else root / args.source_manifest
    )
    output = args.output or root / "reports" / "acl_acceptance" / args.resolution / "stage_performance.json"
    output = output if output.is_absolute() else root / output
    log_root = output.parent / "stage_benchmark_logs"
    benchmark_binary = resolve_binary(root, args.build_dir, args.benchmark_binary)
    if not benchmark_binary.exists():
        raise SystemExit(f"missing benchmark binary: {benchmark_binary}")
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("--warmup must be non-negative and --iterations must be positive")

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
            str(benchmark_binary),
            "--manifest",
            str(manifest_path),
            "--stage",
            name,
            "--device",
            str(args.device),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
        ]
        return_code, stdout, stderr = run_command(command, root, log_path)
        metrics = parse_key_values(stdout)
        avg_ms = metrics.get("avg_ms")
        results.append(
            {
                "stage": name,
                "return_code": return_code,
                "status": "passed" if return_code == 0 else "failed",
                "metrics": metrics,
                "avg_ms": avg_ms if isinstance(avg_ms, (int, float)) else None,
                "log": str(log_path.relative_to(root)),
                "stderr_tail": stderr.strip().splitlines()[-8:],
            }
        )
        if return_code != 0:
            break

    avg_values = [
        result["avg_ms"] for result in results
        if isinstance(result.get("avg_ms"), (int, float))
    ]
    passed = (
        all(result["return_code"] == 0 for result in results)
        and len(results) == len(requested)
        and len(results) == source_count
        and bool(avg_values)
        and all(value > 0 for value in avg_values)
    )
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "passed" if passed else "failed",
        "gate": "stage_performance",
        "resolution": args.resolution,
        "manifest": str(manifest_path),
        "source_manifest": str(source_manifest),
        "stage_count": source_count,
        "manifest_stage_count": acl_count,
        "benchmarked_stage_count": len(results),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "metrics": {
            "avg_ms_min": min(avg_values) if avg_values else 0.0,
            "avg_ms_max": max(avg_values) if avg_values else 0.0,
        },
        "results": results,
    }
    write_report(output, report)
    print(f"wrote {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
