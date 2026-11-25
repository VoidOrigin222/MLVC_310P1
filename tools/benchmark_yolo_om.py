#!/usr/bin/env python3
"""Run a static YOLO OM, validate raw output and decode boxes in source coordinates."""

import argparse
import json
from pathlib import Path
import time

import acl
import numpy as np


def check(status, operation):
    if status != 0:
        raise RuntimeError("{} failed: {}".format(operation, status))


def detections(raw, metadata, confidence=0.25, iou=0.45):
    predictions = raw[0].T.astype(np.float32)
    classes = predictions[:, 4:].argmax(axis=1)
    scores = predictions[:, 4:].max(axis=1)
    selected = np.flatnonzero(scores >= confidence)
    selected = selected[np.argsort(scores[selected])[::-1]][:3000]
    boxes = predictions[:, :4].copy()
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    kept = []
    while selected.size and len(kept) < 100:
        index = selected[0]
        kept.append(int(index))
        rest = selected[1:]
        intersection = np.maximum(0, np.minimum(boxes[index, 2:], boxes[rest, 2:]) -
                                  np.maximum(boxes[index, :2], boxes[rest, :2])).prod(axis=1)
        areas = np.maximum(0, boxes[rest, 2:] - boxes[rest, :2]).prod(axis=1)
        area = np.maximum(0, boxes[index, 2:] - boxes[index, :2]).prod()
        overlap = intersection / np.maximum(area + areas - intersection, 1e-6)
        selected = rest[(classes[rest] != classes[index]) | (overlap <= iou)]
    prep = metadata["preprocess"]
    gain = prep["resize_gain"]
    left, top = prep["padding_ltrb"][:2]
    source_w, source_h = prep["source_wh"]
    result = []
    for index in kept:
        x1, y1, x2, y2 = boxes[index]
        x1, x2 = np.clip((np.array([x1, x2]) - left) / gain, 0, source_w)
        y1, y2 = np.clip((np.array([y1, y2]) - top) / gain, 0, source_h)
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(classes[index])
        result.append({"class_id": class_id, "label": metadata["class_names"][class_id],
                       "conf": float(scores[index]), "cx": float((x1 + x2) / 2),
                       "cy": float((y1 + y2) / 2), "w": float(x2 - x1), "h": float(y2 - y1)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    check(acl.init(), "acl.init")
    check(acl.rt.set_device(0), "set_device")
    context, ret = acl.rt.create_context(0)
    check(ret, "create_context")
    stream, ret = acl.rt.create_stream()
    check(ret, "create_stream")
    model_id, ret = acl.mdl.load_from_file(str(args.model))
    check(ret, "load_from_file")
    desc = acl.mdl.create_desc()
    check(acl.mdl.get_desc(desc, model_id), "get_desc")
    datasets, buffers, pointers = [], [], []
    try:
        if acl.mdl.get_num_inputs(desc) != 1 or acl.mdl.get_num_outputs(desc) != 1:
            raise RuntimeError("expected one input and one raw output")
        input_bytes = args.input.read_bytes()
        input_size = acl.mdl.get_input_size_by_index(desc, 0)
        output_size = acl.mdl.get_output_size_by_index(desc, 0)
        if input_size != len(input_bytes):
            raise RuntimeError("input byte count does not match the OM descriptor")
        for size in (input_size, output_size):
            pointer, ret = acl.rt.malloc(size, 0)
            check(ret, "malloc")
            pointers.append(pointer)
            buffer = acl.create_data_buffer(pointer, size)
            buffers.append(buffer)
            dataset = acl.mdl.create_dataset()
            datasets.append(dataset)
            _, ret = acl.mdl.add_dataset_buffer(dataset, buffer)
            check(ret, "add_dataset_buffer")
        check(acl.rt.memcpy(pointers[0], input_size, acl.util.bytes_to_ptr(input_bytes),
                            input_size, 1), "copy input")
        timings = []
        for index in range(args.iterations + 5):
            start = time.perf_counter()
            check(acl.mdl.execute_async(model_id, datasets[0], datasets[1], stream), "execute_async")
            check(acl.rt.synchronize_stream(stream), "synchronize_stream")
            if index >= 5:
                timings.append((time.perf_counter() - start) * 1000)
        host, ret = acl.rt.malloc_host(output_size)
        check(ret, "malloc_host")
        try:
            check(acl.rt.memcpy(host, output_size, pointers[1], output_size, 2), "copy output")
            output_bytes = acl.util.ptr_to_bytes(host, output_size)
        finally:
            acl.rt.free_host(host)
        dtype = acl.mdl.get_output_data_type(desc, 0)
        if dtype != 1:
            raise RuntimeError("expected FP16 OM output, got dtype {}".format(dtype))
        shape = metadata["output"]["shape"]
        actual = np.frombuffer(output_bytes, dtype=np.float16).reshape(shape).astype(np.float32)
        reference = np.fromfile(args.reference, dtype=np.float16).reshape(shape).astype(np.float32)
        if not np.isfinite(actual).all():
            raise RuntimeError("OM inference returned non-finite predictions")
        error = np.abs(actual - reference)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        binary = args.output.with_suffix(".bin")
        binary.write_bytes(output_bytes)
        report = {
            "model": str(args.model), "input_bytes": input_size, "output_bytes": output_size,
            "iterations": args.iterations, "inference_mean_ms": float(np.mean(timings)),
            "inference_min_ms": float(np.min(timings)), "inference_max_ms": float(np.max(timings)),
            "inference_only_fps": 1000.0 / float(np.mean(timings)),
            "raw_output_finite": True, "box_max_abs_error_px": float(error[:, :4].max()),
            "score_max_abs_error": float(error[:, 4:].max()),
            "detections": detections(actual, metadata),
            "reference_detections": detections(reference, metadata),
            "timing_boundary": "model execution and stream synchronization; excludes preprocessing/postprocessing",
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for buffer in buffers:
            acl.destroy_data_buffer(buffer)
        for dataset in datasets:
            acl.mdl.destroy_dataset(dataset)
        for pointer in pointers:
            acl.rt.free(pointer)
        acl.mdl.destroy_desc(desc)
        acl.mdl.unload(model_id)
        acl.rt.destroy_stream(stream)
        acl.rt.destroy_context(context)
        acl.rt.reset_device(0)
        acl.finalize()


if __name__ == "__main__":
    main()
