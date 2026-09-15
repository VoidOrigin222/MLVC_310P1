#!/usr/bin/env python3
"""Run short MLVC encode/decode baselines and write curated reports."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


PROFILES = ("pipeline-v1",)
RESOLUTIONS = {
    "720p": {
        "manifest": "models/720p_acl/manifest.json",
        "input": "test_video/60s.mp4",
    },
    "1080p": {
        "manifest": "../mlvc1080p/manifest.json",
        "input": "test_video/60s_1080p.mp4",
    },
}
MANIFEST_OVERRIDE_ARGS = {
    "720p": "manifest_720p",
    "1080p": "manifest_1080p",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def parse_key_values(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for raw_line in text.replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            metrics[key] = value
    return metrics


def coerce_value(value: str) -> Any:
    if re.fullmatch(r"-?[0-9]+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][-+]?[0-9]+)?", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def typed_metrics(metrics: dict[str, str]) -> dict[str, Any]:
    return {key: coerce_value(value) for key, value in metrics.items()}


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


def summarize_trace(root: Path, trace_path: Path) -> dict[str, Any]:
    summary_script = root / "tools" / "summarize_profile_trace.py"
    if not trace_path.exists() or not summary_script.exists():
        return {}
    output_path = trace_path.with_suffix(".summary.json")
    completed = subprocess.run(
        [sys.executable, str(summary_script.relative_to(root)), str(trace_path.relative_to(root)),
         "--output", str(output_path.relative_to(root))],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip() or completed.stdout.strip()}
    data = json.loads(output_path.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    return traces[0] if traces else {}


def write_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def selected_manifest(args: argparse.Namespace, resolution: str, root: Path) -> str:
    spec = RESOLUTIONS[resolution]
    override = getattr(args, MANIFEST_OVERRIDE_ARGS[resolution])
    manifest = Path(override or spec["manifest"])
    manifest_path = manifest if manifest.is_absolute() else root / manifest
    if not manifest_path.exists():
        hint = f"--manifest-{resolution} <path>"
        raise SystemExit(
            f"missing ACL manifest for {resolution}: {manifest_path}\n"
            f"Generate the final ACL package first, or pass an explicit manifest with {hint}."
        )
    return relative_to_root(root, manifest_path.resolve())


def ascend_info(root: Path) -> str:
    executable = shutil.which("npu-smi")
    if executable is None:
        return "unavailable"
    completed = subprocess.run(
        [executable, "info"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return completed.stderr.strip() or "unavailable"
    return completed.stdout.strip()


def run_baselines(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    report_root = root / args.output
    artifact_root = report_root / "artifacts"
    config_root = report_root / "run_configs"
    log_root = report_root / "logs"
    build_dir = root / args.build_dir
    encode_binary = build_dir / "mlvc_encode"
    decode_binary = build_dir / "mlvc_decode"
    if not encode_binary.exists() or not decode_binary.exists():
        raise SystemExit(f"missing build binaries under {build_dir}")

    selected_resolutions = args.resolution or ["1080p"]
    selected_profiles = args.profile or list(PROFILES)
    selected_manifests = {
        resolution: selected_manifest(args, resolution, root)
        for resolution in selected_resolutions
    }
    generated_at = utc_now()
    results: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "frames": args.frames,
        "warmup_frames": args.warmup_frames,
        "build_dir": str(build_dir.relative_to(root)),
        "model_manifests": selected_manifests,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ascend": ascend_info(root),
            "cann_home": os.environ.get("CANN_HOME", ""),
        },
        "runs": [],
        "correctness": [],
        "notes": [
            "Short baseline uses three frames by default to cover I-frame, P-reset frame, and normal P-frame.",
            "Raw bitstreams, decoded videos, configs, and logs are generated under reports/performance_baseline and are intentionally ignored unless force-added.",
        ],
    }

    for resolution in selected_resolutions:
        if resolution not in RESOLUTIONS:
            raise SystemExit(f"unknown resolution: {resolution}")
        spec = RESOLUTIONS[resolution]
        for profile in selected_profiles:
            if profile not in PROFILES:
                raise SystemExit(f"unknown profile: {profile}")
            run_id = f"{resolution}_{profile}"
            bitstream = artifact_root / resolution / f"{profile}.bin"
            decoded_dir = artifact_root / resolution / f"{profile}_decoded_png"
            encode_config = config_root / f"encode_{run_id}.toml"
            decode_config = config_root / f"decode_{run_id}.toml"
            encode_trace = report_root / "traces" / f"encode_{run_id}.trace.json"
            decode_trace = report_root / "traces" / f"decode_{run_id}.trace.json"
            write_config(
                encode_config,
                textwrap.dedent(
                    f"""\
                    mode = "encode"
                    input = "{spec['input']}"
                    output = "{bitstream.relative_to(root)}"
                    qp = 17
                    device = {args.device}
                    gop = 128
                    reset_interval = 16
                    frame_num = {args.frames}
                    profile_warmup_frames = {args.warmup_frames}
                    fast = true
                    force_zero_thres = 0.12
                    execution_profile = "{profile}"
                    profile_output = "{(report_root / 'traces' / f'encode_{run_id}.trace.json').relative_to(root)}"

                    [model]
                    manifest = "{selected_manifests[resolution]}"
                    """
                ),
            )
            write_config(
                decode_config,
                textwrap.dedent(
                    f"""\
                    mode = "decode"
                    input = "{bitstream.relative_to(root)}"
                    output = "{decoded_dir.relative_to(root)}"
                    device = {args.device}
                    frame_num = {args.frames}
                    fps = 30
                    format = "png"
                    bitrate = "none"
                    preset = "medium"
                    execution_profile = "{profile}"
                    profile_output = "{(report_root / 'traces' / f'decode_{run_id}.trace.json').relative_to(root)}"

                    [model]
                    manifest = "{selected_manifests[resolution]}"
                    """
                ),
            )
            if decoded_dir.exists():
                shutil.rmtree(decoded_dir)
            bitstream.unlink(missing_ok=True)
            for trace in (encode_trace, decode_trace):
                trace.unlink(missing_ok=True)
            bitstream.parent.mkdir(parents=True, exist_ok=True)
            encode_rc, encode_stdout, encode_stderr = run_command(
                [str(encode_binary), "--config", str(encode_config.relative_to(root))],
                root,
                log_root / f"encode_{run_id}.log",
            )
            decode_rc = 1
            decode_stdout = ""
            decode_stderr = ""
            if encode_rc == 0:
                decode_rc, decode_stdout, decode_stderr = run_command(
                    [str(decode_binary), "--config", str(decode_config.relative_to(root))],
                    root,
                    log_root / f"decode_{run_id}.log",
                )
            encode_metrics = typed_metrics(parse_key_values(encode_stdout))
            decode_metrics = typed_metrics(parse_key_values(decode_stdout))
            run_record: dict[str, Any] = {
                "resolution": resolution,
                "profile": profile,
                "encode": {
                    "return_code": encode_rc,
                    "stdout_metrics": encode_metrics,
                    "stderr_tail": encode_stderr.strip().splitlines()[-5:],
                    "log": str((log_root / f"encode_{run_id}.log").relative_to(root)),
                    "bitstream": str(bitstream.relative_to(root)),
                    "bitstream_bytes": bitstream.stat().st_size if bitstream.exists() else 0,
                    "bitstream_sha256": sha256_file(bitstream) if bitstream.exists() else "",
                    "trace": str(encode_trace.relative_to(root)),
                    "trace_summary": summarize_trace(root, encode_trace),
                },
                "decode": {
                    "return_code": decode_rc,
                    "stdout_metrics": decode_metrics,
                    "stderr_tail": decode_stderr.strip().splitlines()[-5:],
                    "log": str((log_root / f"decode_{run_id}.log").relative_to(root)),
                    "output": str(decoded_dir.relative_to(root)),
                    "output_sha256": sha256_tree(decoded_dir),
                    "trace": str(decode_trace.relative_to(root)),
                    "trace_summary": summarize_trace(root, decode_trace),
                },
            }
            results["runs"].append(run_record)

    for resolution in selected_resolutions:
        baseline = next(
            (
                run
                for run in results["runs"]
                if run["resolution"] == resolution and run["profile"] == "pipeline-v1"
            ),
            None,
        )
        if baseline is None:
            continue
        for run in results["runs"]:
            if run["resolution"] != resolution:
                continue
            results["correctness"].append(
                {
                    "resolution": resolution,
                    "profile": run["profile"],
                    "bitstream_matches_reference": run["encode"]["bitstream_sha256"]
                    == baseline["encode"]["bitstream_sha256"],
                    "decoded_png_matches_reference": run["decode"]["output_sha256"]
                    == baseline["decode"]["output_sha256"],
                    "reference_profile": baseline["profile"],
                    "reference_bitstream_sha256": baseline["encode"]["bitstream_sha256"],
                    "reference_decoded_png_sha256": baseline["decode"]["output_sha256"],
                }
            )
    return results


def write_reports(results: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Performance Baseline Summary",
        "",
        "## English Summary",
        "",
        f"- Generated at: `{results['generated_at']}`",
        f"- Frames per run: `{results['frames']}`",
        f"- Encoder warmup frames before timed/profiled run: `{results.get('warmup_frames', 0)}`",
        f"- Ascend: `{results['environment'].get('ascend', '')}`",
        "- Scope: short reproducible ACL baseline covering I-frame, P-reset frame, normal P-frame, 720p, and 1080p two-part routing.",
        "- Raw logs and artifacts are kept under this report directory for local inspection; curated metrics are in `metrics.json`.",
        "",
        "## 中文摘要",
        "",
        f"- 生成时间：`{results['generated_at']}`",
        f"- 每次运行帧数：`{results['frames']}`",
        f"- 编码正式运行前 warmup 帧数：`{results.get('warmup_frames', 0)}`",
        f"- Ascend：`{results['environment'].get('ascend', '')}`",
        "- 范围：ACL 短基线，覆盖 I 帧、P-reset 帧、普通 P 帧、720p，以及 1080p 的 two-part 路径。",
        "- 原始日志和临时产物保存在本报告目录中供本地检查；可提交的机器可读指标在 `metrics.json`。",
        "",
        "## Model Manifests",
        "",
        "| Resolution | Manifest |",
        "| --- | --- |",
    ]
    for resolution, manifest in results.get("model_manifests", {}).items():
        lines.append(f"| {resolution} | `{manifest}` |")
    lines.extend(
        [
            "",
        "## Metrics",
        "",
        "| Resolution | Profile | Encode FPS | Decode FPS | I MAD | Bitstream Bytes | Encode RC | Decode RC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in results["runs"]:
        encode = run["encode"]["stdout_metrics"]
        decode = run["decode"]["stdout_metrics"]
        lines.append(
            "| {resolution} | {profile} | {encode_fps} | {decode_fps} | {mad} | {bytes} | {erc} | {drc} |".format(
                resolution=run["resolution"],
                profile=run["profile"],
                encode_fps=encode.get("encode_fps", ""),
                decode_fps=decode.get("decode_fps", ""),
                mad=encode.get("i_encode_decode_mad", ""),
                bytes=run["encode"].get("bitstream_bytes", ""),
                erc=run["encode"].get("return_code", ""),
                drc=run["decode"].get("return_code", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Correctness Checks",
            "",
            "| Resolution | Profile | Bitstream matches reference | Decoded PNG matches reference |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in results["correctness"]:
        lines.append(
            f"| {check['resolution']} | {check['profile']} | {check['bitstream_matches_reference']} | {check['decoded_png_matches_reference']} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "source scripts/acl_env.sh",
            "cmake --build build -j2",
            f"python3 tools/run_performance_baseline.py --frames {results['frames']} --warmup-frames {results.get('warmup_frames', 0)}",
            "```",
            "",
            "## Notes / 说明",
            "",
            "- English: these are correctness-oriented short baselines, not final throughput claims. Full-video numbers should be regenerated after trace output and materialization telemetry land.",
            "- 中文：这些数据是偏正确性验证的短基线，不作为最终吞吐量结论。trace output 与 materialization telemetry 完成后，应重新生成完整视频性能数据。",
            "",
        ]
    )
    (output_root / "baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def baseline_passed(results: dict[str, Any]) -> bool:
    runs = results.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return False
    for run in runs:
        if not isinstance(run, dict):
            return False
        encode = run.get("encode")
        decode = run.get("decode")
        if not isinstance(encode, dict) or not isinstance(decode, dict):
            return False
        if encode.get("return_code") != 0 or decode.get("return_code") != 0:
            return False
    correctness = results.get("correctness", [])
    if not isinstance(correctness, list) or not correctness:
        return False
    for check in correctness:
        if not isinstance(check, dict):
            return False
        if check.get("bitstream_matches_reference") is not True:
            return False
        if check.get("decoded_png_matches_reference") is not True:
            return False
    return True


def export_acceptance_reports(metrics_path: Path, output_root: Path,
                              resolutions: list[str]) -> None:
    import export_acl_acceptance

    metrics_path = metrics_path.resolve(strict=False)
    metrics = export_acl_acceptance.load_json(metrics_path)
    ok = True
    for resolution in resolutions:
        end_to_end = export_acl_acceptance.build_end_to_end_report(metrics, resolution)
        performance = export_acl_acceptance.build_performance_report(metrics, resolution)
        end_to_end["metrics_source"] = str(metrics_path)
        performance["metrics_source"] = str(metrics_path)
        export_acl_acceptance.write_json(
            output_root / resolution / "end_to_end_correctness.json", end_to_end
        )
        export_acl_acceptance.write_json(output_root / resolution / "performance.json", performance)
        ok = ok and end_to_end["status"] == "passed" and performance["status"] == "passed"
    if not ok:
        raise SystemExit("acceptance export failed; inspect reports/acl_acceptance")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--build-dir", default="build", help="build directory containing binaries")
    parser.add_argument("--output", default="reports/performance_baseline", help="report directory")
    parser.add_argument("--frames", type=int, default=3, help="frames per run")
    parser.add_argument("--warmup-frames", type=int, default=0,
                        help="encoder-only profile_warmup_frames before the measured run")
    parser.add_argument("--device", type=int, default=0, help="Ascend device id")
    parser.add_argument("--resolution", action="append", choices=sorted(RESOLUTIONS))
    parser.add_argument("--manifest-720p", help="override the default 720p ACL manifest")
    parser.add_argument("--manifest-1080p", help="override the default 1080p ACL manifest")
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument(
        "--export-acceptance",
        action="store_true",
        help="Export end-to-end correctness and performance evidence from generated metrics.",
    )
    parser.add_argument(
        "--acceptance-output-root",
        default="reports/acl_acceptance",
        help="Output directory used with --export-acceptance.",
    )
    args = parser.parse_args()
    if args.frames < 3:
        raise SystemExit("--frames must be at least 3 to cover I, P-reset, and normal P frames")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")
    root = Path(args.root).resolve()
    results = run_baselines(args)
    output_root = root / args.output
    write_reports(results, output_root)
    if args.export_acceptance:
        export_acceptance_reports(
            output_root / "metrics.json",
            root / args.acceptance_output_root,
            args.resolution or list(RESOLUTIONS),
        )
    return 0 if baseline_passed(results) else 1


if __name__ == "__main__":
    sys.exit(main())
