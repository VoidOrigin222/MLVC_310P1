#!/usr/bin/env python3
"""Export ACL end-to-end and performance acceptance evidence from baseline metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def resolution_runs(metrics: dict[str, Any], resolution: str) -> list[dict[str, Any]]:
    runs = metrics.get("runs")
    if not isinstance(runs, list):
        raise ValueError("baseline metrics missing runs array")
    return [
        run for run in runs
        if isinstance(run, dict) and run.get("resolution") == resolution
    ]


def correctness_for(metrics: dict[str, Any], resolution: str) -> list[dict[str, Any]]:
    checks = metrics.get("correctness", [])
    if not isinstance(checks, list):
        return []
    return [
        check for check in checks
        if isinstance(check, dict) and check.get("resolution") == resolution
    ]


def build_end_to_end_report(metrics: dict[str, Any], resolution: str) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run in resolution_runs(metrics, resolution):
        encode = run.get("encode", {})
        decode = run.get("decode", {})
        if not isinstance(encode, dict) or not isinstance(decode, dict):
            raise ValueError(f"{resolution} run missing encode/decode objects")
        runs.append(
            {
                "profile": run.get("profile", ""),
                "encode_return_code": encode.get("return_code"),
                "decode_return_code": decode.get("return_code"),
                "bitstream_bytes": encode.get("bitstream_bytes", 0),
                "bitstream_sha256": encode.get("bitstream_sha256", ""),
                "decoded_output_sha256": decode.get("output_sha256", ""),
                "encode_log": encode.get("log", ""),
                "decode_log": decode.get("log", ""),
            }
        )
    checks: list[dict[str, Any]] = []
    checks.append({"name": "has_runs", "ok": bool(runs)})
    for run in runs:
        checks.append(
            {
                "name": f"{run['profile']}.encode_decode_return_codes",
                "ok": run["encode_return_code"] == 0 and run["decode_return_code"] == 0,
            }
        )
        checks.append(
            {
                "name": f"{run['profile']}.bitstream_nonempty",
                "ok": isinstance(run["bitstream_bytes"], int) and run["bitstream_bytes"] > 0,
            }
        )
        checks.append(
            {
                "name": f"{run['profile']}.decoded_output_present",
                "ok": bool(run["decoded_output_sha256"]),
            }
        )
    for check in correctness_for(metrics, resolution):
        profile = check.get("profile", "")
        checks.append(
            {
                "name": f"{profile}.bitstream_reference_check",
                "ok": check.get("bitstream_matches_reference") is True,
            }
        )
        checks.append(
            {
                "name": f"{profile}.decoded_reference_check",
                "ok": check.get("decoded_png_matches_reference") is True,
            }
        )
    passed = all(check["ok"] for check in checks)
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "passed" if passed else "failed",
        "gate": "end_to_end_correctness",
        "resolution": resolution,
        "metrics_source": "",
        "runs": runs,
        "checks": checks,
    }


def build_performance_report(metrics: dict[str, Any], resolution: str) -> dict[str, Any]:
    encode_fps: list[float] = []
    decode_fps: list[float] = []
    for run in resolution_runs(metrics, resolution):
        encode = run.get("encode", {})
        decode = run.get("decode", {})
        if isinstance(encode, dict):
            value = number(encode.get("stdout_metrics", {}).get("encode_fps")
                           if isinstance(encode.get("stdout_metrics"), dict) else None)
            if value is not None:
                encode_fps.append(value)
        if isinstance(decode, dict):
            value = number(decode.get("stdout_metrics", {}).get("decode_fps")
                           if isinstance(decode.get("stdout_metrics"), dict) else None)
            if value is not None:
                decode_fps.append(value)
    report_metrics = {
        "encode_fps_min": min(encode_fps) if encode_fps else 0.0,
        "encode_fps_max": max(encode_fps) if encode_fps else 0.0,
        "decode_fps_min": min(decode_fps) if decode_fps else 0.0,
        "decode_fps_max": max(decode_fps) if decode_fps else 0.0,
    }
    checks = [
        {"name": "has_encode_fps", "ok": bool(encode_fps)},
        {"name": "has_decode_fps", "ok": bool(decode_fps)},
        {"name": "encode_fps_positive", "ok": report_metrics["encode_fps_min"] > 0.0},
        {"name": "decode_fps_positive", "ok": report_metrics["decode_fps_min"] > 0.0},
    ]
    passed = all(check["ok"] for check in checks)
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "passed" if passed else "failed",
        "gate": "performance",
        "resolution": resolution,
        "metrics_source": "",
        "stage_performance_report": "stage_performance.json",
        "metrics": report_metrics,
        "checks": checks,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("reports/performance_baseline/metrics.json"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/acl_acceptance"))
    parser.add_argument("--resolution", action="append", choices=("720p", "1080p"))
    args = parser.parse_args()

    metrics_path = args.metrics.resolve(strict=False)
    metrics = load_json(metrics_path)
    resolutions = args.resolution or sorted(
        {
            run.get("resolution")
            for run in metrics.get("runs", [])
            if isinstance(run, dict) and isinstance(run.get("resolution"), str)
        }
    )
    if not resolutions:
        raise SystemExit("no resolutions found in baseline metrics")

    ok = True
    for resolution in resolutions:
        end_to_end = build_end_to_end_report(metrics, resolution)
        performance = build_performance_report(metrics, resolution)
        end_to_end["metrics_source"] = str(metrics_path)
        performance["metrics_source"] = str(metrics_path)
        write_json(args.output_root / resolution / "end_to_end_correctness.json", end_to_end)
        write_json(args.output_root / resolution / "performance.json", performance)
        ok = ok and end_to_end["status"] == "passed" and performance["status"] == "passed"
        print(
            f"{resolution}: end_to_end={end_to_end['status']} performance={performance['status']}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
