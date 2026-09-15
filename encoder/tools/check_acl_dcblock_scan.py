#!/usr/bin/env python3
"""Validate the recorded DCBlock rewrite scan report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_STAGE_COUNTS = {
    "i_encoder": 7,
    "i_hyper_encoder": 3,
    "i_hyper_decoder_prior": 6,
    "i_spatial_prior": 12,
    "i_decoder": 14,
    "p_reference_frame_adaptor": 1,
    "p_reference_feature_adaptor": 0,
    "p_reference_context": 6,
    "p_analysis_encoder": 3,
    "p_hyper_encoder": 3,
    "p_hyper_temporal_prior": 7,
    "p_spatial_prior": 2,
    "p_synthesis_decoder": 3,
    "p_reconstruction": 4,
    "i_spatial_prior_decode_init": 0,
    "i_spatial_prior_decode_step_1": 4,
    "i_spatial_prior_decode_step_2": 4,
    "i_spatial_prior_decode_step_3": 4,
    "p_spatial_prior_decode_init": 0,
    "p_spatial_prior_decode_step": 2,
}


def package_resolution(record: dict[str, Any]) -> str:
    manifest = record.get("manifest")
    if not isinstance(manifest, str):
        raise ValueError("scan package missing manifest path")
    parts = Path(manifest).parts
    for part in ("720p", "1080p"):
        if part in parts:
            return part
    raise ValueError(f"cannot infer resolution from manifest path: {manifest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, default=Path("reports/acl_dcblock_scan.json"))
    args = parser.parse_args()

    try:
        packages = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(packages, list):
            raise ValueError("scan report must be a list")
        seen_resolutions: set[str] = set()
        expected_total = sum(EXPECTED_STAGE_COUNTS.values())
        for package in packages:
            if not isinstance(package, dict):
                raise ValueError("scan package is not an object")
            resolution = package_resolution(package)
            seen_resolutions.add(resolution)
            total = package.get("total")
            if total != expected_total:
                raise ValueError(
                    f"{resolution} total mismatch: expected {expected_total}, got {total}"
                )
            models = package.get("models")
            if not isinstance(models, list):
                raise ValueError(f"{resolution} models must be a list")
            actual = {
                item.get("name"): item.get("post_sigmoid_chunk_add")
                for item in models
                if isinstance(item, dict)
            }
            if actual != EXPECTED_STAGE_COUNTS:
                raise ValueError(f"{resolution} stage count map mismatch")
            if sum(actual.values()) != total:
                raise ValueError(f"{resolution} stage counts do not sum to total")
        if seen_resolutions != {"720p", "1080p"}:
            raise ValueError(f"scan report resolutions mismatch: {sorted(seen_resolutions)}")
        print(f"acl_dcblock_scan status=ok total={expected_total} resolutions=720p,1080p")
        return 0
    except Exception as error:
        print(f"check_acl_dcblock_scan failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
