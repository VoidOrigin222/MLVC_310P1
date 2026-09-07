#!/usr/bin/env python3
"""Summarize MLVC Chrome trace JSON into copy/stage telemetry."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("traceEvents", [])


def summarize(path: Path) -> dict[str, Any]:
    events = load_trace(path)
    duration_by_category: dict[str, float] = defaultdict(float)
    count_by_category: Counter[str] = Counter()
    copy_bytes: Counter[str] = Counter()
    copy_counts: Counter[str] = Counter()
    skipped_copy_bytes: Counter[str] = Counter()
    skipped_copy_counts: Counter[str] = Counter()
    stage_wall_time_ms: dict[str, float] = defaultdict(float)
    sync_count = 0
    sync_wait_ms = 0.0
    cpu_entropy_ms = 0.0
    device_copy_ms = 0.0
    video_read_ms = 0.0
    video_write_ms = 0.0
    bitstream_io_ms = 0.0
    hot_path_allocation_count = 0
    hot_path_allocation_bytes = 0
    pipeline_node_counts: Counter[str] = Counter()
    pipeline_node_time_ms: dict[str, float] = defaultdict(float)
    pipeline_packet_acquire_count = 0
    pipeline_packet_release_count = 0

    for event in events:
        phase = event.get("ph")
        category = event.get("cat", "")
        args = event.get("args", {})
        if phase == "X":
            duration_ms = float(event.get("dur", 0.0)) / 1000.0
            count_by_category[category] += 1
            duration_by_category[category] += duration_ms
            name = event.get("name", "")
            if category == "acl":
                stage_wall_time_ms[name] += duration_ms
            elif category == "copy":
                if args.get("aggregate") is True:
                    device_copy_ms += duration_ms
                    continue
                direction = str(args.get("direction", "unknown"))
                skipped = "skip_reason" in args or str(event.get("name", "")).startswith("copy.d2h.skip.")
                try:
                    bytes_value = int(args.get("bytes", 0))
                except (TypeError, ValueError):
                    bytes_value = 0
                if skipped:
                    skipped_copy_counts[direction] += 1
                    skipped_copy_bytes[direction] += bytes_value
                else:
                    device_copy_ms += duration_ms
                    copy_counts[direction] += 1
                    copy_bytes[direction] += bytes_value
            elif category == "sync":
                sync_count += 1
                sync_wait_ms += duration_ms
            elif category == "entropy":
                cpu_entropy_ms += duration_ms
            elif category == "video_io":
                if str(name).startswith("video.read") or str(name).startswith("video.synthetic"):
                    video_read_ms += duration_ms
                elif str(name).startswith("video.write"):
                    video_write_ms += duration_ms
            elif category == "bitstream_io":
                bitstream_io_ms += duration_ms
            elif category == "pipeline":
                node_type = str(args.get("node_type", "unknown"))
                pipeline_node_counts[node_type] += 1
                pipeline_node_time_ms[node_type] += duration_ms
                if name == "pipeline.packet.acquire":
                    pipeline_packet_acquire_count += 1
                elif name == "pipeline.packet.release":
                    pipeline_packet_release_count += 1
        elif phase == "C":
            name = str(event.get("name", ""))
            try:
                if name == "allocation.hot_path.count":
                    hot_path_allocation_count += int(args.get("count", 0))
                elif name == "allocation.hot_path.bytes":
                    hot_path_allocation_bytes += int(args.get("bytes", 0))
            except (TypeError, ValueError):
                pass

    return {
        "trace": str(path),
        "event_count": len(events),
        "duration_by_category_ms": dict(sorted((k, round(v, 6)) for k, v in duration_by_category.items())),
        "count_by_category": dict(sorted(count_by_category.items())),
        "d2h_bytes": int(copy_bytes.get("D2H", 0)),
        "h2d_bytes": int(copy_bytes.get("H2D", 0)),
        "d2h_count": int(copy_counts.get("D2H", 0)),
        "h2d_count": int(copy_counts.get("H2D", 0)),
        "skipped_d2h_bytes": int(skipped_copy_bytes.get("D2H", 0)),
        "skipped_h2d_bytes": int(skipped_copy_bytes.get("H2D", 0)),
        "skipped_d2h_count": int(skipped_copy_counts.get("D2H", 0)),
        "skipped_h2d_count": int(skipped_copy_counts.get("H2D", 0)),
        "stream_sync_count": int(sync_count),
        "cpu_entropy_time_ms": round(cpu_entropy_ms, 6),
        "cpu_wait_time_ms": round(sync_wait_ms, 6),
        "device_copy_time_ms": round(device_copy_ms, 6),
        "video_read_time_ms": round(video_read_ms, 6),
        "video_write_time_ms": round(video_write_ms, 6),
        "bitstream_io_time_ms": round(bitstream_io_ms, 6),
        "stage_wall_time_ms": dict(sorted((k, round(v, 6)) for k, v in stage_wall_time_ms.items())),
        "hot_path_allocation_count": hot_path_allocation_count,
        "hot_path_allocation_bytes": hot_path_allocation_bytes,
        "pipeline_node_counts": dict(sorted(pipeline_node_counts.items())),
        "pipeline_node_time_ms": dict(sorted((k, round(v, 6)) for k, v in pipeline_node_time_ms.items())),
        "pipeline_packet_acquire_count": pipeline_packet_acquire_count,
        "pipeline_packet_release_count": pipeline_packet_release_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.traces]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"traces": summaries}, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
