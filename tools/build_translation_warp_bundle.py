#!/usr/bin/env python3
"""Add the verified frame-reference OM to a separate motion manifest."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_record(root, record):
    path = (root / record["file"]).resolve()
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"bundle artifact size or SHA-256 mismatch: {path}")


def build(args):
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    output = args.output.resolve() if args.output else root / "manifest_motion_ascend310p1.json"
    if output == manifest_path or output.parent != root:
        raise ValueError("motion manifest must be a separate file in the original bundle directory")
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("runtime") != "acl" or source.get("soc_version") != "Ascend310P1":
        raise ValueError("expected an ACL Ascend310P1 source manifest")
    if source.get("dtype") not in ("fp16", "float16"):
        raise ValueError("source bundle must use FP16")
    for record in [source["sidecar"], *source.get("runtime_artifacts", []), *source["models"]]:
        verify_record(root, record)
    if any(record["name"] == "MLVCReferenceFromFrame" for record in source["models"]):
        raise ValueError("source manifest already contains frame-reference stage")
    encoder = next(record for record in source["models"] if record["name"] == "MLVCEncoder")
    input_spec = next(spec for spec in encoder["inputs"] if spec["name"] == "x")
    feature_spec = next(spec for spec in encoder["inputs"] if spec["name"] == "ref_feature")
    batch, channels, height, width = input_spec["shape"]
    if (batch != 1 or channels != 3 or height <= 0 or width <= 0 or height % 8 or width % 8 or
            input_spec["dtype"] not in ("fp16", "float16") or
            feature_spec["dtype"] not in ("fp16", "float16") or
            feature_spec["shape"] != [1, 96, height // 8, width // 8]):
        raise ValueError("source bundle is not the expected 48 feature + 48 memory model")
    om = args.om.resolve()
    relative_om = om.relative_to(root).as_posix()
    actual_sha = sha256(om)
    if len(args.expected_sha) != 64 or actual_sha != args.expected_sha.lower():
        raise ValueError("reference OM differs from the explicitly expected SHA-256")
    if om.suffix != ".om" or om.stat().st_size == 0:
        raise ValueError("reference stage must be a nonempty .om file")
    motion = copy.deepcopy(source)
    motion["models"].append({
        "name": "MLVCReferenceFromFrame", "file": relative_om, "backend": "acl",
        "bytes": om.stat().st_size, "sha256": actual_sha,
        "inputs": [{"name": "ref_frame", "dtype": "float16", "shape": [1, 3, height, width]}],
        "outputs": [{"name": "ref_feature", "dtype": "float16", "shape": feature_spec["shape"]}],
    })
    content = json.dumps(motion, indent=2) + "\n"
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != motion:
            raise ValueError("different motion manifest already exists; refusing to replace it")
    else:
        output.write_text(content, encoding="utf-8")
    print(json.dumps({"manifest": str(output), "manifest_sha256": sha256(output),
                      "stage_sha256": actual_sha, "stage_bytes": om.stat().st_size,
                      "original_models_preserved": len(source["models"])}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
