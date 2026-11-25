#!/usr/bin/env python3
"""Copy a verified codec bundle into the ulbvc deployment naming scheme."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if destination.exists():
        parser.error("destination must be new")
    manifest_path = source / "manifest_motion_ascend310p1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prefixes = {re.sub(r"(?:Encoder|Decoder)$", "", item["name"])
                for item in manifest["models"]
                if item["name"].endswith(("Encoder", "Decoder"))}
    if len(prefixes) != 1:
        parser.error("expected one codec prefix with Encoder and Decoder roles")
    prefix = prefixes.pop()
    pattern = re.compile(re.escape(prefix), re.IGNORECASE)

    def rename(text):
        def replacement(match):
            value = match.group()
            return "ULBVC" if value.isupper() else "ulbvc" if value.islower() else "Ulbvc"
        return pattern.sub(replacement, text)

    records = [manifest["sidecar"], *manifest["runtime_artifacts"], *manifest["models"]]
    for record in records:
        file = (source / record["file"]).resolve()
        if not file.is_relative_to(source):
            parser.error("bundle paths must stay inside the source directory")
        data = file.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("source bundle integrity mismatch: " + record["file"])
    for file in source.rglob("*"):
        if not file.is_file():
            continue
        target = destination / rename(file.relative_to(source).as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        if file.suffix == ".json":
            target.write_text(rename(file.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copyfile(file, target)
    converted_path = destination / rename(manifest_path.name)
    converted = json.loads(converted_path.read_text(encoding="utf-8"))
    for record in [converted["sidecar"], *converted["runtime_artifacts"], *converted["models"]]:
        data = (destination / record["file"]).read_bytes()
        record["bytes"] = len(data)
        record["sha256"] = hashlib.sha256(data).hexdigest()
    converted_path.write_text(json.dumps(converted, indent=2) + "\n", encoding="utf-8")
    print(converted_path)


if __name__ == "__main__":
    main()
