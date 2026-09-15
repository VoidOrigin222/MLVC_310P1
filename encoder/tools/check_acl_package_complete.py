#!/usr/bin/env python3
"""Check ACL package stage coverage against the source model manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_names(manifest: dict[str, Any], label: str) -> list[str]:
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{label} missing non-empty models array")
    names: list[str] = []
    seen: set[str] = set()
    for index, record in enumerate(models):
        if not isinstance(record, dict):
            raise ValueError(f"{label} models[{index}] is not an object")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} models[{index}] missing name")
        if name in seen:
            raise ValueError(f"{label} duplicated model name: {name}")
        seen.add(name)
        names.append(name)
    return names


def check_ordered_subset(source_names: list[str], acl_names: list[str]) -> None:
    source_index = {name: index for index, name in enumerate(source_names)}
    unknown = [name for name in acl_names if name not in source_index]
    if unknown:
        raise ValueError("ACL manifest contains stages not present in source manifest: " + ", ".join(unknown))
    actual_indexes = [source_index[name] for name in acl_names]
    if actual_indexes != sorted(actual_indexes):
        raise ValueError("ACL manifest stage order does not follow source manifest order")


def check_stage_coverage(source_manifest: Path, acl_manifest: Path, allow_partial: bool) -> tuple[int, int, str]:
    source_names = model_names(load_json(source_manifest), "source manifest")
    acl_names = model_names(load_json(acl_manifest), "ACL manifest")
    check_ordered_subset(source_names, acl_names)
    if not allow_partial and acl_names != source_names:
        missing = [name for name in source_names if name not in set(acl_names)]
        extra = [name for name in acl_names if name not in set(source_names)]
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError("ACL manifest is not complete: " + " ".join(details))
    mode = "partial" if allow_partial else "complete"
    return len(acl_names), len(source_names), mode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--acl-manifest", required=True, type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow ACL manifest to contain an ordered subset of source stages.",
    )
    args = parser.parse_args()

    try:
        acl_count, source_count, mode = check_stage_coverage(
            args.source_manifest, args.acl_manifest, args.allow_partial
        )
        print(f"acl_package_complete status=ok mode={mode} stages={acl_count}/{source_count}")
        return 0
    except Exception as error:
        print(f"check_acl_package_complete failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
