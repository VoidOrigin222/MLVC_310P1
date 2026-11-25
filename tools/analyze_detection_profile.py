#!/usr/bin/env python3
"""Summarize host timing and model-call overlap from an encoder Chrome trace.

Host API overlap does not establish concurrent device kernel execution.
Use the accompanying CANN task timeline for device concurrency.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(values)
    return {"count": len(values), "mean_ms": statistics.mean(values),
            "p50_ms": statistics.median(values),
            "p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))],
            "max_ms": values[-1]}


def analyze(path):
    events = [event for event in json.loads(path.read_text())["traceEvents"]
              if event.get("ph") == "X"]
    grouped = defaultdict(list)
    for event in events:
        name = event["name"]
        stage = event.get("args", {}).get("stage")
        if event.get("cat") == "acl" and stage:
            name = "model." + stage
        grouped[name].append(event["dur"] / 1000)
    detection = sorted((event for event in events if event["name"] == "detection.model_call"),
                       key=lambda event: event["ts"])
    codec = sorted((event for event in events if event.get("cat") == "acl" and
                    event.get("args", {}).get("stage") == "ULBVCEncoder"),
                   key=lambda event: event["ts"])
    overlaps = []
    for detected in detection:
        frame_id = detected["args"]["frame_id"]
        if frame_id >= len(codec):
            continue
        encoded = codec[frame_id]
        overlap = max(0, min(detected["ts"] + detected["dur"], encoded["ts"] + encoded["dur"])
                      - max(detected["ts"], encoded["ts"]))
        overlaps.append(overlap / 1000)
    preparation = defaultdict(list)
    for event in events:
        if event["name"] in {"detection.color_convert", "detection.resize", "detection.pack_fp16"}:
            preparation[event["args"]["frame_id"]].append(event)
    preparation_windows = merge_intervals([
        (min(event["ts"] for event in group), max(event["ts"] + event["dur"] for event in group))
        for group in preparation.values()])
    model_windows = merge_intervals([(event["ts"], event["ts"] + event["dur"]) for event in detection])
    return {"trace": str(path),
            "timing_boundary": "host wall times; nested/concurrent times must not be summed",
            "stages": {name: distribution(values) for name, values in sorted(grouped.items())},
            "preparation_model_host_overlap_ms": intersection_duration(preparation_windows, model_windows) / 1000,
            "same_frame_host_model_call_overlap": {
                "frames": len(overlaps), "frames_with_overlap": sum(value > 0 for value in overlaps),
                "mean_overlap_ms": statistics.mean(overlaps) if overlaps else None,
                "device_kernel_concurrency": "requires CANN task timeline"}}


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def intersection_duration(first, second):
    i = j = 0
    duration = 0
    while i < len(first) and j < len(second):
        duration += max(0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return duration


def analyze_cann(folder, detector_model_id, codec_model_id):
    timeline = json.loads(next(folder.glob("msprof_*.json")).read_text())
    detector_threads = {event["tid"] for event in timeline
                        if event.get("name") == "AscendCL@aclmdlExecuteAsync"}
    if len(detector_threads) != 1:
        raise ValueError("expected exactly one asynchronous detector thread")
    thread = detector_threads.pop()
    calls = defaultdict(list)
    for event in timeline:
        if event.get("ph") == "X" and event.get("tid") == thread:
            calls[event["name"]].append(event)
    api_stats = {name: distribution([event["dur"] / 1000 for event in events])
                 for name, events in calls.items()
                 if name in {"AscendCL@aclrtMemcpy", "AscendCL@aclmdlExecuteAsync",
                             "AscendCL@aclrtSynchronizeStream"}}
    copies = sorted(calls["AscendCL@aclrtMemcpy"], key=lambda event: float(event["ts"]))
    executions = len(calls["AscendCL@aclmdlExecuteAsync"])
    if len(copies) != executions * 2:
        raise ValueError("expected one H2D input and one D2H output copy per detector execution")
    rows = list(csv.DictReader(next(folder.glob("step_trace_*.csv")).open()))
    windows, model_stats = {}, {}
    for model_id in (codec_model_id, detector_model_id):
        steps = [row for row in rows if row["Model ID"] == model_id]
        model_stats[model_id] = distribution([float(row["Iteration Time(us)"]) / 1000 for row in steps])
        windows[model_id] = merge_intervals([
            (float(row["Iteration End(us)"]) - float(row["Iteration Time(us)"]),
             float(row["Iteration End(us)"])) for row in steps])
    return {"detector_model_id": detector_model_id, "codec_model_id": codec_model_id,
            "detector_thread": thread, "detector_api": api_stats,
            "detector_h2d_host_mean_ms": statistics.mean(event["dur"] / 1000 for event in copies[::2]),
            "detector_d2h_host_mean_ms": statistics.mean(event["dur"] / 1000 for event in copies[1::2]),
            "device_model_execution_windows": model_stats,
            "device_model_window_overlap_ms": intersection_duration(
                windows[codec_model_id], windows[detector_model_id]) / 1000,
            "interpretation": "model execution windows include scheduling and operator gaps; overlap is not AI Core utilization"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann-folder", type=Path)
    parser.add_argument("--detector-model-id", default="4")
    parser.add_argument("--codec-model-id", default="1")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"profiles": [analyze(path) for path in args.traces]}
    if args.cann_folder:
        result["cann"] = analyze_cann(args.cann_folder, args.detector_model_id, args.codec_model_id)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
