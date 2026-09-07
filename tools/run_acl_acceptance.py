#!/usr/bin/env python3
"""Run the ACL final-manifest acceptance sequence for selected resolutions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], root: Path, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def manifest_for(models_root: Path, resolution: str) -> Path:
    return models_root / f"{resolution}_acl" / "manifest.json"


def source_manifest_for(source_root: Path, resolution: str) -> Path:
    return source_root / resolution / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source-root", type=Path, default=Path("/root/workspace/cgc/mlvc_cpp/models"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--acceptance-root", type=Path, default=Path("reports/acl_acceptance"))
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--resolution", action="append", choices=("720p", "1080p"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--stage-warmup", type=int, default=50)
    parser.add_argument("--stage-iterations", type=int, default=2000)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=0.005)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run after manifest existence checks.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    source_root = args.source_root if args.source_root.is_absolute() else root / args.source_root
    models_root = args.models_root if args.models_root.is_absolute() else root / args.models_root
    acceptance_root = (
        args.acceptance_root if args.acceptance_root.is_absolute() else root / args.acceptance_root
    )
    resolutions = args.resolution or ["720p", "1080p"]
    python = sys.executable

    for resolution in resolutions:
        manifest = manifest_for(models_root, resolution)
        source_manifest = source_manifest_for(source_root, resolution)
        if not manifest.is_file():
            raise SystemExit(f"missing final ACL manifest: {manifest}")
        if not source_manifest.is_file():
            raise SystemExit(f"missing source manifest: {source_manifest}")
        run(
            [
                python,
                "tools/verify_models.py",
                str(manifest),
            ],
            root,
            args.dry_run,
        )
        run(
            [
                python,
                "tools/check_acl_package_complete.py",
                "--source-manifest",
                str(source_manifest),
                "--acl-manifest",
                str(manifest),
            ],
            root,
            args.dry_run,
        )
        run(
            [
                python,
                "tools/run_acl_stage_validation.py",
                "--resolution",
                resolution,
                "--manifest",
                str(manifest),
                "--source-manifest",
                str(source_manifest),
                "--output",
                str(acceptance_root / resolution / "single_stage_correctness.json"),
                "--build-dir",
                str(args.build_dir),
                "--device",
                str(args.device),
                "--atol",
                str(args.atol),
                "--rtol",
                str(args.rtol),
            ],
            root,
            args.dry_run,
        )
        run(
            [
                python,
                "tools/run_acl_stage_benchmark.py",
                "--resolution",
                resolution,
                "--manifest",
                str(manifest),
                "--source-manifest",
                str(source_manifest),
                "--output",
                str(acceptance_root / resolution / "stage_performance.json"),
                "--build-dir",
                str(args.build_dir),
                "--device",
                str(args.device),
                "--warmup",
                str(args.stage_warmup),
                "--iterations",
                str(args.stage_iterations),
            ],
            root,
            args.dry_run,
        )

    if not args.skip_end_to_end:
        baseline_command = [
            python,
            "tools/run_performance_baseline.py",
            "--build-dir",
            str(args.build_dir),
            "--device",
            str(args.device),
            "--frames",
            str(args.frames),
            "--warmup-frames",
            str(args.warmup_frames),
            "--export-acceptance",
            "--acceptance-output-root",
            str(acceptance_root),
        ]
        for resolution in resolutions:
            baseline_command.extend(["--resolution", resolution])
            manifest = manifest_for(models_root, resolution)
            if resolution == "720p":
                baseline_command.extend(["--manifest-720p", str(manifest)])
            elif resolution == "1080p":
                baseline_command.extend(["--manifest-1080p", str(manifest)])
        run(baseline_command, root, args.dry_run)

    run(
        [
            python,
            "tools/acl_goal_readiness.py",
            "--source-root",
            str(source_root),
            "--models-root",
            str(models_root),
            "--acceptance-root",
            str(acceptance_root),
            "--require-complete",
        ],
        root,
        args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
