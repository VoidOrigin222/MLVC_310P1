#!/usr/bin/env python3
"""Package current local sources and codec assets with a SHA256 file inventory."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--changed-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = set()
    names = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root).decode().split("\0")
    for name in names:
        if name and (root / name).is_file():
            files.add(name)
    for path in (root / "models/codec").rglob("*"):
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    detector_root = root / "models/detection/yolo11s"
    for path in (detector_root / "fp16_opset17").rglob("*"):
        if path.is_file() and (path.suffix in {".om", ".json", ".txt"}
                               or path.name.endswith(".om.sha256")):
            files.add(path.relative_to(root).as_posix())
    if (detector_root / "SOURCE.json").is_file():
        files.add((detector_root / "SOURCE.json").relative_to(root).as_posix())
    records = []
    for name in sorted(files):
        data = (root / name).read_bytes()
        records.append({"file": name, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory = args.output.parent / "DEPLOYMENT_SOURCE.json"
    previous = {}
    if args.changed_only and inventory.is_file():
        previous = {record["file"]: record["sha256"]
                    for record in json.loads(inventory.read_text())["files"]}
    inventory.write_text(json.dumps({"files": records}, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(args.output, "w:gz") as bundle:
        for record in records:
            if previous.get(record["file"]) == record["sha256"]:
                continue
            bundle.add(root / record["file"], arcname=record["file"], recursive=False)
        bundle.add(inventory, arcname=inventory.name)
    print(f"Packaged {len(records)} files: {args.output}")


if __name__ == "__main__":
    main()
