#!/usr/bin/env python3
"""Report ACL package conversion status without invoking ATC or AOE."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_stage_files(source_manifest: Path) -> list[tuple[str, str]]:
    manifest = load_json(source_manifest)
    if manifest is None:
        return []
    stages: list[tuple[str, str]] = []
    for record in manifest.get("models", []):
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        file_value = record.get("file")
        stem = Path(file_value).stem if isinstance(file_value, str) and file_value else f"{name}.sim"
        stages.append((name, stem))
    return stages


def artifact_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        info["mtime"] = stat.st_mtime
        info["mtime_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime))
        if path.is_file():
            info["bytes"] = stat.st_size
    return info


def expected_stage_artifacts(package_root: Path, stem: str) -> dict[str, dict[str, Any]]:
    return {
        "original_onnx": artifact_info(package_root / "onnx_original" / f"{stem}.onnx"),
        "optimized_onnx": artifact_info(package_root / "onnx_optimized" / f"{stem}.onnx"),
        "rewrite_report": artifact_info(package_root / "reports" / "rewrite" / f"{stem}.json"),
        "atc_om": artifact_info(package_root / "om_atc" / f"{stem}.om"),
        "atc_io_json": artifact_info(package_root / "reports" / "io" / f"{stem}.atc.json"),
        "aoe_om": artifact_info(package_root / "om_aoe" / f"{stem}.om"),
        "aoe_io_json": artifact_info(package_root / "reports" / "io" / f"{stem}.aoe.json"),
    }


def read_proc_command(pid_dir: Path) -> str | None:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def read_proc_ppid(pid_dir: Path) -> str:
    try:
        for line in (pid_dir / "status").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("PPid:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def read_proc_elapsed_seconds(pid_dir: Path, now_ticks: int, ticks_per_second: int) -> int | None:
    try:
        stat_text = (pid_dir / "stat").read_text(encoding="utf-8", errors="replace")
        fields = stat_text.rsplit(") ", 1)[1].split()
        start_ticks = int(fields[19])
    except (OSError, IndexError, ValueError):
        return None
    return max(0, int((now_ticks - start_ticks) / ticks_per_second))


def format_elapsed(seconds: int | None) -> str:
    if seconds is None:
        return ""
    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    prefix = f"{days}d" if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{secs:02d}"


def command_kind(command: str) -> str | None:
    basename = Path(command.split()[0]).name if command.split() else ""
    if basename == "aoe":
        return "aoe"
    if basename == "atc":
        return "atc"
    if "convert_acl_models.py" in command and "--audit-only" not in command:
        return "convert_acl_models"
    return None


def parse_command_value(command: str, key: str) -> str:
    prefix = f"{key}="
    parts = command.split()
    for index, part in enumerate(parts):
        if part.startswith(prefix):
            return part[len(prefix) :]
        if part == key and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def active_processes(repo_root: Path) -> list[dict[str, Any]]:
    proc_root = Path("/proc")
    try:
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    except (ValueError, KeyError):
        ticks_per_second = 100
    try:
        now_ticks = int(float((proc_root / "uptime").read_text().split()[0]) * ticks_per_second)
    except OSError:
        now_ticks = 0

    processes: list[dict[str, Any]] = []
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        if int(pid_dir.name) == os.getpid():
            continue
        command = read_proc_command(pid_dir)
        if command is None:
            continue
        kind = command_kind(command)
        if kind is None:
            continue
        model = parse_command_value(command, "--model")
        output = parse_command_value(command, "--output")
        stage = Path(output).name or Path(model).stem
        if stage.endswith(".sim"):
            stage = stage[:-4]
        processes.append(
            {
                "pid": int(pid_dir.name),
                "ppid": read_proc_ppid(pid_dir),
                "kind": kind,
                "elapsed_seconds": read_proc_elapsed_seconds(pid_dir, now_ticks, ticks_per_second),
                "elapsed": format_elapsed(read_proc_elapsed_seconds(pid_dir, now_ticks, ticks_per_second)),
                "stage": stage,
                "model": model,
                "output": output,
                "in_repo": str(repo_root) in command,
                "command": command,
            }
        )
    return sorted(processes, key=lambda item: (item["kind"], item["pid"]))


def package_status(
    resolution: str,
    source_manifest: Path,
    acl_root: Path,
    now: float,
    stale_seconds: int,
) -> dict[str, Any]:
    package_root = acl_root / f"{resolution}_acl"
    stages = manifest_stage_files(source_manifest)
    complete: list[str] = []
    incomplete: list[str] = []
    missing_by_stage: dict[str, list[str]] = {}
    stage_artifacts: dict[str, dict[str, Any]] = {}
    for name, stem in stages:
        artifacts = expected_stage_artifacts(package_root, stem)
        missing = [key for key, info in artifacts.items() if not info["exists"]]
        stage_artifacts[name] = artifacts
        if missing:
            incomplete.append(name)
            missing_by_stage[name] = missing
        else:
            complete.append(name)

    audit = load_json(package_root / "reports" / "package_audit.json")
    newest_aoe_update = 0.0
    newest_aoe_path = ""
    aoe_reports = package_root / "reports" / "aoe"
    if aoe_reports.exists():
        for path in aoe_reports.rglob("*"):
            try:
                if path.is_file():
                    mtime = path.stat().st_mtime
                else:
                    continue
            except OSError:
                continue
            if mtime > newest_aoe_update:
                newest_aoe_update = mtime
                newest_aoe_path = str(path)

    return {
        "resolution": resolution,
        "source_manifest": str(source_manifest),
        "package_root": str(package_root),
        "stage_count": len(stages),
        "tuned_stage_count": len(complete),
        "remaining_stage_count": len(incomplete),
        "tuned_stages": complete,
        "remaining_stages": incomplete,
        "missing_by_stage": missing_by_stage,
        "manifest_partial": artifact_info(package_root / "manifest.partial.json"),
        "manifest_final": artifact_info(package_root / "manifest.json"),
        "package_audit": artifact_info(package_root / "reports" / "package_audit.json"),
        "audit_summary": audit.get("summary") if isinstance(audit, dict) else None,
        "newest_aoe_update": {
            "path": newest_aoe_path,
            "age_seconds": None if newest_aoe_update == 0.0 else max(0, int(now - newest_aoe_update)),
            "fresh": newest_aoe_update != 0.0 and now - newest_aoe_update <= stale_seconds,
        },
        "stage_artifacts": stage_artifacts,
    }


def print_human(report: dict[str, Any]) -> None:
    for package in report["packages"]:
        print(
            f"{package['resolution']}: tuned={package['tuned_stage_count']}/"
            f"{package['stage_count']} remaining={package['remaining_stage_count']} "
            f"final_manifest={package['manifest_final']['exists']}"
        )
        if package["remaining_stages"]:
            print("  remaining: " + ", ".join(package["remaining_stages"]))
        newest = package["newest_aoe_update"]
        if newest["path"]:
            freshness = "fresh" if newest["fresh"] else "stale"
            print(f"  newest_aoe_update: {freshness} age={newest['age_seconds']}s path={newest['path']}")
    processes = report["active_processes"]
    if not processes:
        print("active_conversion_processes: none")
        return
    print(f"active_conversion_processes: {len(processes)}")
    for process in processes:
        print(
            f"  pid={process['pid']} kind={process['kind']} stage={process['stage']} "
            f"elapsed={process['elapsed']} in_repo={process['in_repo']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/root/workspace/cgc/mlvc_cpp/models"))
    parser.add_argument("--acl-root", type=Path, default=Path("models"))
    parser.add_argument("--resolution", action="append", choices=("720p", "1080p"))
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    now = time.time()
    resolutions = args.resolution or ["720p", "1080p"]
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "stale_seconds": args.stale_seconds,
        "packages": [
            package_status(
                resolution,
                args.source_root / resolution / "manifest.json",
                args.acl_root,
                now,
                args.stale_seconds,
            )
            for resolution in resolutions
        ],
        "active_processes": active_processes(repo_root),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
