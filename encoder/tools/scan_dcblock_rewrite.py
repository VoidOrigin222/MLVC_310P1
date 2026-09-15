#!/usr/bin/env python3
"""Scan MLVC ONNX model packages for PostSigmoidChunkAdd rewrite points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rewrite_wsiluchunkadd_onnx import rewrite_model  # noqa: E402


def model_file(record: dict[str, Any], manifest_dir: Path) -> Path:
    file_name = record.get("file") or f"{record['name']}.sim.onnx"
    path = Path(file_name)
    return path if path.is_absolute() else manifest_dir / path


def count_matches(path: Path) -> int:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    _, matches = rewrite_model(model, "postsigmoidchunkadd")
    return len(matches)


def scan_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent
    stages: list[dict[str, Any]] = []
    total = 0
    for record in manifest.get("models", []):
        path = model_file(record, manifest_dir)
        count = count_matches(path)
        stages.append(
            {
                "name": record["name"],
                "file": str(path),
                "post_sigmoid_chunk_add": count,
            }
        )
        total += count
    return {"manifest": str(manifest_path), "total": total, "models": stages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    reports = [scan_manifest(path) for path in args.manifest]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for report in reports:
        print(f"{report['manifest']}: total={report['total']}")
        for item in report["models"]:
            print(f"  {item['name']}: {item['post_sigmoid_chunk_add']}")


if __name__ == "__main__":
    main()
