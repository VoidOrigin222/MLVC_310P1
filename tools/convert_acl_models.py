#!/usr/bin/env python3
"""Convert a MLVC ONNX model manifest to rewritten ONNX, ATC OM, AOE OM, and ACL manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if TYPE_CHECKING:
    import onnx

ONNX_DTYPE_NAMES = {
    1: "float32",
    10: "float16",
    3: "int8",
    5: "int16",
    6: "int32",
    2: "uint8",
}

AOE_WORKSPACE_RETENTION = {
    "policy": "preserve",
    "reason": "AOE workspaces and tuning caches may speed up later retries.",
}


def require_onnx_modules() -> tuple[Any, Any]:
    import onnx
    from onnx import shape_inference

    return onnx, shape_inference


def require_rewrite_model() -> Any:
    from rewrite_wsiluchunkadd_onnx import rewrite_model

    return rewrite_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(
    command: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        if log.tell() > 0:
            log.write("\n\n")
        log.write("# started_at=" + dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
        if cwd is not None:
            log.write("# cwd=" + str(cwd) + "\n")
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=cwd,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit {result.returncode}: {' '.join(command)}")


def capture_command(command: list[str], max_chars: int = 12000) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    return result.stdout[:max_chars]


def tool_metadata(atc: str, aoe: str) -> dict[str, str]:
    return {
        "atc": atc,
        "aoe": aoe,
        "atc_version": capture_command([atc, "--version"]),
        "aoe_version": capture_command([aoe, "--version"]),
        "atc_help": capture_command([atc, "--help"]),
        "aoe_help": capture_command([aoe, "--help"]),
    }


def dtype_name(elem_type: int) -> str:
    if elem_type not in ONNX_DTYPE_NAMES:
        raise RuntimeError(f"unsupported ONNX input dtype: {elem_type}")
    return ONNX_DTYPE_NAMES[elem_type]


def tensor_specs(path: Path, kind: str) -> list[dict[str, Any]]:
    onnx, shape_inference = require_onnx_modules()
    model = onnx.load(path)
    inferred = shape_inference.infer_shapes(model)
    values = inferred.graph.input if kind == "input" else inferred.graph.output
    specs: list[dict[str, Any]] = []
    initializer_names = {init.name for init in inferred.graph.initializer}
    for value in values:
        if kind == "input" and value.name in initializer_names:
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            raise RuntimeError(f"{path}: tensor has no shape: {value.name}")
        shape: list[int] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            else:
                raise RuntimeError(f"{path}: dynamic dimension on tensor {value.name}")
        specs.append({"name": value.name, "dtype": dtype_name(tensor_type.elem_type), "shape": shape})
    return specs


def input_shape_arg(inputs: list[dict[str, Any]]) -> str:
    return ";".join(f"{item['name']}:{','.join(str(v) for v in item['shape'])}" for item in inputs)


def input_fp16_nodes_arg(inputs: list[dict[str, Any]]) -> str:
    return ";".join(item["name"] for item in inputs if item["dtype"] == "float16")


def acl_runtime_output_specs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return runtime output specs matching --output_type=FP16 OM outputs."""
    runtime_outputs: list[dict[str, Any]] = []
    for output in outputs:
        runtime_output = dict(output)
        runtime_output["dtype"] = "float16"
        runtime_outputs.append(runtime_output)
    return runtime_outputs


def relative_to(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def resolve_tool(name: str, cann_home: Path) -> str:
    candidate = cann_home / "bin" / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"cannot find required tool: {name}")
    return found


def aoe_level(mode: str) -> str:
    if mode == "highest":
        return "O1"
    if mode in {"O1", "O2"}:
        return mode
    raise RuntimeError(f"unsupported --aoe-mode: {mode}")


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        info["kind"] = "directory" if path.is_dir() else "file"
        if path.is_file():
            info["bytes"] = path.stat().st_size
            info["sha256"] = sha256(path)
    return info


def select_records(manifest: dict[str, Any], requested_stages: set[str]) -> list[dict[str, Any]]:
    records = [
        record
        for record in manifest.get("models", [])
        if not requested_stages or record.get("name") in requested_stages
    ]
    if requested_stages:
        found = {record["name"] for record in records}
        missing = sorted(requested_stages - found)
        if missing:
            raise RuntimeError("unknown --stage values: " + ", ".join(missing))
    return records


def merge_manifest_records(
    output_root: Path,
    source_manifest: dict[str, Any],
    new_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Merge newly converted stages into manifest.partial.json in source manifest order."""
    partial_path = output_root / "manifest.partial.json"
    records_by_name: dict[str, dict[str, Any]] = {}
    if partial_path.is_file():
        existing = json.loads(partial_path.read_text(encoding="utf-8"))
        for record in existing.get("models", []):
            name = record.get("name")
            if isinstance(name, str) and name:
                records_by_name[name] = record
    for record in new_manifest.get("models", []):
        name = record.get("name")
        if isinstance(name, str) and name:
            records_by_name[name] = record

    source_order = [
        record.get("name")
        for record in source_manifest.get("models", [])
        if isinstance(record.get("name"), str)
    ]
    merged = dict(new_manifest)
    merged["models"] = [records_by_name[name] for name in source_order if name in records_by_name]
    return merged


def build_acl_manifest_record_from_artifacts(
    record: dict[str, Any],
    source_root: Path,
    output_root: Path,
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Build one ACL manifest record from already-generated local artifacts."""
    name = record["name"]
    src_file = Path(record.get("file") or f"{name}.sim.onnx")
    source_onnx = src_file if src_file.is_absolute() else source_root / src_file
    original_onnx = dirs["onnx_original"] / source_onnx.name
    optimized_onnx = dirs["onnx_optimized"] / source_onnx.name
    rewrite_report = dirs["rewrite"] / f"{source_onnx.stem}.json"
    atc_om = Path(str(dirs["om_atc"] / source_onnx.stem) + ".om")
    aoe_om = Path(str(dirs["om_aoe"] / source_onnx.stem) + ".om")
    atc_io_json = dirs["io"] / f"{source_onnx.stem}.atc.json"
    aoe_io_json = dirs["io"] / f"{source_onnx.stem}.aoe.json"
    required = (original_onnx, optimized_onnx, rewrite_report, atc_om, aoe_om, atc_io_json, aoe_io_json)
    if any(not path.is_file() for path in required):
        return None

    report = json.loads(rewrite_report.read_text(encoding="utf-8"))
    match_count = report.get("match_count")
    if not isinstance(match_count, int):
        raise RuntimeError(f"{rewrite_report}: missing integer match_count")

    new_record: dict[str, Any] = {
        "name": name,
        "file": relative_to(aoe_om, output_root),
        "onnx_file": relative_to(original_onnx, output_root),
        "optimized_onnx_file": relative_to(optimized_onnx, output_root),
        "atc_om_file": relative_to(atc_om, output_root),
        "backend": "acl",
        "bytes": aoe_om.stat().st_size,
        "sha256": sha256(aoe_om),
        "frame_type": record.get("frame_type", ""),
        "route": record.get("route", ""),
        "inputs": tensor_specs(optimized_onnx, "input"),
        "outputs": acl_runtime_output_specs(tensor_specs(optimized_onnx, "output")),
        "optimizations": {
            "post_sigmoid_chunk_add": match_count,
            "fp16_io": True,
            "aoe": args.aoe_mode,
            "aoe_tune_optimization_level": aoe_level(args.aoe_mode),
        },
    }
    if "replaces" in record:
        new_record["replaces"] = record["replaces"]
    return new_record


def rebuild_partial_manifest_from_artifacts(
    manifest: dict[str, Any],
    source_root: Path,
    output_root: Path,
    dirs: dict[str, Path],
    requested_stages: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Write manifest.partial.json for all stages whose required artifacts are present."""
    sidecar = manifest["sidecar"]
    source_sidecar = source_root / sidecar["file"]
    target_sidecar = output_root / Path(sidecar["file"]).name
    if not target_sidecar.is_file():
        shutil.copy2(source_sidecar, target_sidecar)

    partial_manifest: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime": "acl",
        "soc_version": args.soc,
        "dtype": manifest.get("dtype", "fp16"),
        "sidecar": {
            "file": target_sidecar.name,
            "bytes": target_sidecar.stat().st_size,
            "sha256": sha256(target_sidecar),
        },
        "models": [],
    }
    for record in select_records(manifest, requested_stages):
        rebuilt = build_acl_manifest_record_from_artifacts(record, source_root, output_root, dirs, args)
        if rebuilt is not None:
            partial_manifest["models"].append(rebuilt)

    partial_path = output_root / "manifest.partial.json"
    if not partial_manifest["models"]:
        if partial_path.exists():
            partial_path.unlink()
        print(
            json.dumps(
                {
                    "partial_manifest": str(partial_path),
                    "rebuilt_stage_count": 0,
                    "written": False,
                },
                sort_keys=True,
            )
        )
        return partial_manifest

    partial_path.write_text(json.dumps(partial_manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "partial_manifest": str(partial_path),
                "rebuilt_stage_count": len(partial_manifest["models"]),
                "written": True,
            },
            sort_keys=True,
        )
    )
    return partial_manifest


def active_conversion_processes() -> list[dict[str, str]]:
    proc_root = Path("/proc")
    current_pid = os.getpid()
    matches: list[dict[str, str]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        command = " ".join(argv)
        if "codex-linux-sandbox" in command or "bwrap --new-session" in command:
            continue
        basename = Path(argv[0]).name if argv else ""
        has_converter_script = any(Path(part).name == "convert_acl_models.py" for part in argv[1:])
        is_match = (
            basename in {"atc", "aoe"}
            or (
                (basename == "convert_acl_models.py" or basename.startswith("python"))
                and has_converter_script
                and "--audit-only" not in argv
            )
        )
        if not is_match:
            continue
        matches.append({"pid": str(pid), "cmd": command})
    return sorted(matches, key=lambda item: int(item["pid"]))


def check_no_active_conversion_processes() -> None:
    matches = active_conversion_processes()
    if not matches:
        return
    details = "\n".join(f"  pid={item['pid']} {item['cmd']}" for item in matches[:12])
    extra = "" if len(matches) <= 12 else f"\n  ... {len(matches) - 12} more"
    raise RuntimeError(
        "refusing to start conversion while ATC/AOE/conversion processes are active.\n"
        f"{details}{extra}\n"
        "Wait for the active job to finish, or pass --allow-concurrent-conversion if you "
        "have intentionally isolated the devices and output paths."
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.skip_atc and not args.audit_only:
        raise RuntimeError("ATC is required for final ACL manifests; use --resume to reuse existing ATC artifacts")
    if args.skip_aoe and not args.audit_only:
        raise RuntimeError("AOE is required for final ACL manifests; --skip-aoe is intentionally blocked")


def write_package_audit(
    manifest: dict[str, Any],
    source_root: Path,
    output_root: Path,
    dirs: dict[str, Path],
    requested_stages: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = select_records(manifest, requested_stages)
    audit: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "audit_only",
        "source_manifest": str(args.manifest),
        "source_model_root": str(source_root),
        "output": str(output_root),
        "soc_version": args.soc,
        "aoe_mode": args.aoe_mode,
        "aoe_tune_optimization_level": aoe_level(args.aoe_mode),
        "aoe_workspace_retention": AOE_WORKSPACE_RETENTION,
        "requested_stages": sorted(requested_stages),
        "sidecar": {},
        "manifest": {
            "partial": artifact_info(output_root / "manifest.partial.json"),
            "final": artifact_info(output_root / "manifest.json"),
        },
        "summary": {},
        "models": [],
    }
    sidecar = manifest["sidecar"]
    source_sidecar = source_root / sidecar["file"]
    target_sidecar = output_root / Path(sidecar["file"]).name
    audit["sidecar"] = {
        "source": artifact_info(source_sidecar),
        "target": artifact_info(target_sidecar),
    }

    complete = 0
    missing_by_kind: dict[str, list[str]] = {
        "source_onnx": [],
        "original_onnx": [],
        "optimized_onnx": [],
        "rewrite_report": [],
        "atc_om": [],
        "aoe_om": [],
        "atc_io_json": [],
        "aoe_io_json": [],
    }
    complete_stages: list[str] = []
    incomplete_stages: list[str] = []
    missing_by_stage: dict[str, list[str]] = {}
    for record in records:
        name = record["name"]
        src_file = Path(record.get("file") or f"{name}.sim.onnx")
        source_onnx = src_file if src_file.is_absolute() else source_root / src_file
        original_onnx = dirs["onnx_original"] / source_onnx.name
        optimized_onnx = dirs["onnx_optimized"] / source_onnx.name
        rewrite_report = dirs["rewrite"] / f"{source_onnx.stem}.json"
        atc_om = Path(str(dirs["om_atc"] / source_onnx.stem) + ".om")
        aoe_om = Path(str(dirs["om_aoe"] / source_onnx.stem) + ".om")
        atc_io_json = dirs["io"] / f"{source_onnx.stem}.atc.json"
        aoe_io_json = dirs["io"] / f"{source_onnx.stem}.aoe.json"
        aoe_work_dir = dirs["aoe"] / source_onnx.stem
        artifacts = {
            "source_onnx": artifact_info(source_onnx),
            "original_onnx": artifact_info(original_onnx),
            "optimized_onnx": artifact_info(optimized_onnx),
            "rewrite_report": artifact_info(rewrite_report),
            "atc_om": artifact_info(atc_om),
            "aoe_om": artifact_info(aoe_om),
            "aoe_work_dir": artifact_info(aoe_work_dir),
            "atc_io_json": artifact_info(atc_io_json),
            "aoe_io_json": artifact_info(aoe_io_json),
        }
        required_artifacts = {
            kind: info for kind, info in artifacts.items() if kind != "aoe_work_dir"
        }
        missing = [kind for kind, info in required_artifacts.items() if not info["exists"]]
        for kind in missing:
            missing_by_kind[kind].append(name)
        if not missing:
            complete += 1
            complete_stages.append(name)
        else:
            incomplete_stages.append(name)
            missing_by_stage[name] = missing
        audit["models"].append(
            {
                "name": name,
                "status": "complete" if not missing else "incomplete",
                "missing": missing,
                "artifacts": artifacts,
            }
        )

    audit["summary"] = {
        "stage_count": len(records),
        "complete_stage_count": complete,
        "incomplete_stage_count": len(records) - complete,
        "complete_stages": complete_stages,
        "incomplete_stages": incomplete_stages,
        "missing_by_stage": missing_by_stage,
        "missing_by_kind": {kind: names for kind, names in missing_by_kind.items() if names},
        "final_manifest_ready": complete == len(records) and bool(records),
    }
    reports = dirs["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    audit_path = reports / "package_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(audit_path), **audit["summary"]}, sort_keys=True))
    return audit


def rewrite_one(original: Path, optimized: Path, report: Path, expected_count: int) -> None:
    onnx, _ = require_onnx_modules()
    rewrite_model = require_rewrite_model()
    model = onnx.load(original)
    onnx.checker.check_model(model)
    rewritten, matches = rewrite_model(model, "postsigmoidchunkadd")
    if len(matches) != expected_count:
        raise RuntimeError(f"{original}: expected {expected_count} matches, found {len(matches)}")
    optimized.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewritten, optimized)
    # ONNX's built-in checker cannot validate the default-domain Ascend custom op schema.
    # Load the model back and run shape extraction as a structural check instead.
    _ = tensor_specs(optimized, "input")
    _ = tensor_specs(optimized, "output")
    report.write_text(
        json.dumps(
            {
                "source": str(original),
                "output": str(optimized),
                "op_type": "PostSigmoidChunkAdd",
                "match_count": len(matches),
                "matches": [
                    {
                        "output": str(match["output"]),
                        "source": str(match["source"]),
                        "sigmoid_output": str(match["sigmoid_output"]),
                        "input_shape": match["input_shape"],
                        "output_shape": match["output_shape"],
                    }
                    for match in matches
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--soc", default="Ascend310P3")
    parser.add_argument("--cann-home", type=Path, default=Path(os.environ.get("CANN_HOME", "/usr/local/Ascend/cann")))
    parser.add_argument("--device", default="0")
    parser.add_argument("--aoe-mode", default="highest")
    parser.add_argument(
        "--aoe-aicore-num",
        type=int,
        help="Optional AOE --aicore_num limit for resource-constrained isolated retries.",
    )
    parser.add_argument(
        "--aoe-progress-bar",
        choices=("on", "off"),
        default="off",
        help="AOE progress display mode. Default is off to keep logs compact.",
    )
    parser.add_argument(
        "--aoe-enable-single-stream",
        action="store_true",
        help="Pass --enable_single_stream=true to AOE for resource-constrained retries.",
    )
    parser.add_argument(
        "--aoe-extra-arg",
        action="append",
        default=[],
        help="Additional raw AOE argument. Can be provided multiple times.",
    )
    parser.add_argument(
        "--skip-atc",
        action="store_true",
        help="Only valid with --audit-only. Final manifests require ATC OM and ATC I/O metadata.",
    )
    parser.add_argument("--skip-aoe", action="store_true")
    parser.add_argument(
        "--allow-concurrent-conversion",
        action="store_true",
        help="Bypass the process guard that normally prevents overlapping ATC/AOE conversion jobs.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only inspect expected conversion artifacts and write reports/package_audit.json.",
    )
    parser.add_argument(
        "--rebuild-partial-manifest",
        action="store_true",
        help="With --audit-only, rebuild manifest.partial.json from complete existing artifacts.",
    )
    parser.add_argument("--source-model-root", type=Path)
    parser.add_argument(
        "--stage",
        action="append",
        help="Convert only the named stage. Can be provided multiple times for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing ONNX/ATC/AOE artifacts when present; final manifest still requires AOE OMs.",
    )
    args = parser.parse_args()

    validate_args(args)
    if not args.audit_only and not args.allow_concurrent_conversion:
        check_no_active_conversion_processes()

    manifest = load_manifest(args.manifest)
    source_root = args.source_model_root or args.manifest.parent
    output_root = args.output
    dirs = {
        "onnx_original": output_root / "onnx_original",
        "onnx_optimized": output_root / "onnx_optimized",
        "om_atc": output_root / "om_atc",
        "om_aoe": output_root / "om_aoe",
        "reports": output_root / "reports",
        "rewrite": output_root / "reports" / "rewrite",
        "atc": output_root / "reports" / "atc",
        "aoe": output_root / "reports" / "aoe",
        "io": output_root / "reports" / "io",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    requested_stages = set(args.stage or [])
    if args.audit_only:
        write_package_audit(manifest, source_root, output_root, dirs, requested_stages, args)
        if args.rebuild_partial_manifest:
            rebuild_partial_manifest_from_artifacts(
                manifest, source_root, output_root, dirs, requested_stages, args
            )
        return

    atc = resolve_tool("atc", args.cann_home)
    aoe = resolve_tool("aoe", args.cann_home)
    audit = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "soc_version": args.soc,
        "aoe_mode": args.aoe_mode,
        "aoe_tune_optimization_level": aoe_level(args.aoe_mode),
        "aoe_workspace_retention": AOE_WORKSPACE_RETENTION,
        "aoe_resource_options": {
            "aicore_num": args.aoe_aicore_num,
            "progress_bar": args.aoe_progress_bar,
            "enable_single_stream": args.aoe_enable_single_stream,
            "extra_args": args.aoe_extra_arg,
        },
        "tools": tool_metadata(atc, aoe),
        "models": [],
    }

    sidecar = manifest["sidecar"]
    source_sidecar = source_root / sidecar["file"]
    target_sidecar = output_root / Path(sidecar["file"]).name
    shutil.copy2(source_sidecar, target_sidecar)
    new_manifest: dict[str, Any] = {
        "created_at": audit["created_at"],
        "runtime": "acl",
        "soc_version": args.soc,
        "dtype": manifest.get("dtype", "fp16"),
        "sidecar": {
            "file": target_sidecar.name,
            "bytes": target_sidecar.stat().st_size,
            "sha256": sha256(target_sidecar),
        },
        "models": [],
    }

    failures: list[str] = []
    records = select_records(manifest, requested_stages)

    for record in records:
        name = record["name"]
        src_file = Path(record.get("file") or f"{name}.sim.onnx")
        source_onnx = src_file if src_file.is_absolute() else source_root / src_file
        original_onnx = dirs["onnx_original"] / source_onnx.name
        optimized_onnx = dirs["onnx_optimized"] / source_onnx.name
        rewrite_report = dirs["rewrite"] / f"{source_onnx.stem}.json"
        atc_prefix = dirs["om_atc"] / source_onnx.stem
        atc_om = Path(str(atc_prefix) + ".om")
        aoe_prefix = dirs["om_aoe"] / source_onnx.stem
        aoe_om = Path(str(aoe_prefix) + ".om")
        atc_io_json = dirs["io"] / f"{source_onnx.stem}.atc.json"
        aoe_io_json = dirs["io"] / f"{source_onnx.stem}.aoe.json"

        try:
            if not args.resume or not original_onnx.exists():
                shutil.copy2(source_onnx, original_onnx)
            original_inputs = tensor_specs(original_onnx, "input")
            original_outputs = tensor_specs(original_onnx, "output")
            onnx, _ = require_onnx_modules()
            rewrite_model = require_rewrite_model()
            expected_count = len(rewrite_model(onnx.load(original_onnx), "postsigmoidchunkadd")[1])
            if not args.resume or not optimized_onnx.exists() or not rewrite_report.exists():
                rewrite_one(original_onnx, optimized_onnx, rewrite_report, expected_count)
            optimized_inputs = tensor_specs(optimized_onnx, "input")
            optimized_outputs = tensor_specs(optimized_onnx, "output")
            runtime_outputs = acl_runtime_output_specs(optimized_outputs)

            if not args.skip_atc:
                fp16_nodes = input_fp16_nodes_arg(optimized_inputs)
                if not args.resume or not atc_om.exists():
                    atc_command = [
                        atc,
                        f"--model={optimized_onnx}",
                        "--framework=5",
                        f"--output={atc_prefix}",
                        f"--soc_version={args.soc}",
                        "--input_format=NCHW",
                        f"--input_shape={input_shape_arg(optimized_inputs)}",
                        "--output_type=FP16",
                        "--precision_mode=force_fp16",
                        "--op_select_implmode=high_performance",
                        "--log=info",
                    ]
                    if fp16_nodes:
                        atc_command.append(f"--input_fp16_nodes={fp16_nodes}")
                    run_command(atc_command, dirs["atc"] / f"{source_onnx.stem}.log")
                if not args.resume or not atc_io_json.exists():
                    run_command([atc, "--mode=1", f"--om={atc_om}", f"--json={atc_io_json}"], dirs["atc"] / f"{source_onnx.stem}.io.log")

                aoe_work_dir = dirs["aoe"] / source_onnx.stem
                # Preserve AOE workspaces/caches across runs; later retries may reuse them.
                aoe_work_dir.mkdir(parents=True, exist_ok=True)
                if not args.resume or not aoe_om.exists():
                    aoe_command = [
                        aoe,
                        f"--model={optimized_onnx.resolve()}",
                        "--framework=5",
                        f"--output={aoe_prefix.resolve()}",
                        "--input_format=NCHW",
                        f"--input_shape={input_shape_arg(optimized_inputs)}",
                        "--output_type=FP16",
                        "--job_type=2",
                        f"--device={args.device}",
                        f"--tune_optimization_level={aoe_level(args.aoe_mode)}",
                        f"--progress_bar={args.aoe_progress_bar}",
                        "--precision_mode=force_fp16",
                        "--op_select_implmode=high_performance",
                    ]
                    if args.aoe_aicore_num is not None:
                        aoe_command.append(f"--aicore_num={args.aoe_aicore_num}")
                    if args.aoe_enable_single_stream:
                        aoe_command.append("--enable_single_stream=true")
                    if fp16_nodes:
                        aoe_command.append(f"--input_fp16_nodes={fp16_nodes}")
                    aoe_command.extend(args.aoe_extra_arg)
                    # This CANN AOE version cannot tune an already-generated .om as input; it
                    # expects framework input and emits the tuned OM itself. Keep ATC OM as the
                    # baseline artifact, then make AOE-from-ONNX the mandatory runtime artifact.
                    run_command(aoe_command, dirs["aoe"] / f"{source_onnx.stem}.log",
                                cwd=aoe_work_dir)
                if not aoe_om.exists():
                    raise RuntimeError(f"AOE did not produce expected OM: {aoe_om}")
                if not args.resume or not aoe_io_json.exists():
                    run_command([atc, "--mode=1", f"--om={aoe_om}", f"--json={aoe_io_json}"], dirs["aoe"] / f"{source_onnx.stem}.io.log")

            if not aoe_om.exists():
                raise RuntimeError(f"missing final AOE OM: {aoe_om}")

            new_record: dict[str, Any] = {
                "name": name,
                "file": relative_to(aoe_om, output_root),
                "onnx_file": relative_to(original_onnx, output_root),
                "optimized_onnx_file": relative_to(optimized_onnx, output_root),
                "atc_om_file": relative_to(atc_om, output_root),
                "backend": "acl",
                "bytes": aoe_om.stat().st_size,
                "sha256": sha256(aoe_om),
                "frame_type": record.get("frame_type", ""),
                "route": record.get("route", ""),
                "inputs": optimized_inputs,
                "outputs": runtime_outputs,
                "optimizations": {
                    "post_sigmoid_chunk_add": expected_count,
                    "fp16_io": True,
                    "aoe": args.aoe_mode,
                    "aoe_tune_optimization_level": aoe_level(args.aoe_mode),
                },
            }
            if "replaces" in record:
                new_record["replaces"] = record["replaces"]
            new_manifest["models"].append(new_record)
            audit["models"].append(
                {
                    "name": name,
                    "status": "ok",
                    "post_sigmoid_chunk_add": expected_count,
                    "original_inputs": original_inputs,
                    "original_outputs": original_outputs,
                    "optimized_inputs": optimized_inputs,
                    "optimized_outputs": optimized_outputs,
                    "atc_om": str(atc_om),
                    "aoe_om": str(aoe_om),
                    "aoe_work_dir": str(aoe_work_dir),
                    "aoe_workspace_retention": AOE_WORKSPACE_RETENTION,
                }
            )
            print(json.dumps({"name": name, "status": "ok", "post_sigmoid_chunk_add": expected_count}))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
            audit["models"].append({"name": name, "status": "failed", "error": str(exc)})
            print(json.dumps({"name": name, "status": "failed", "error": str(exc)}), file=sys.stderr)

    (dirs["reports"] / "conversion_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if new_manifest["models"]:
        partial_path = output_root / "manifest.partial.json"
        partial_manifest = (
            merge_manifest_records(output_root, manifest, new_manifest)
            if requested_stages
            else new_manifest
        )
        partial_path.write_text(json.dumps(partial_manifest, indent=2), encoding="utf-8")
        print(f"wrote {partial_path}")
    if failures:
        raise RuntimeError("ACL conversion failed:\n" + "\n".join(failures))
    if not requested_stages:
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
        print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
