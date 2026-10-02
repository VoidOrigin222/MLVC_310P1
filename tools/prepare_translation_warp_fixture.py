#!/usr/bin/env python3
"""Prepare NCHW FP16 YUV444 frames using the sample inference's YUV rules.

Run outside the original Python checkout. Chroma uses nearest upsampling,
each 8-bit sample is normalized by 255, and bottom/right edges replicate.
"""

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def read_plane(plane, height, width):
    rows = np.frombuffer(plane, dtype=np.uint8).reshape(-1, plane.line_size)
    if rows.shape[0] < height or plane.line_size < width:
        raise ValueError("decoded YUV plane is smaller than its declared dimensions")
    return rows[:height, :width].copy()


def prepare_frame(frame, coded_height, coded_width):
    yuv = frame.reformat(format="yuv420p")
    height, width = yuv.height, yuv.width
    if height % 2 or width % 2:
        raise ValueError("YUV420 fixture requires even visible dimensions")
    if coded_height < height or coded_width < width:
        raise ValueError("coded dimensions cannot crop the input video")
    planes = [read_plane(yuv.planes[0], height, width)]
    for plane in yuv.planes[1:3]:
        chroma = read_plane(plane, height // 2, width // 2)
        planes.append(np.repeat(np.repeat(chroma, 2, axis=0), 2, axis=1))
    x = np.stack(planes).astype(np.float32) / np.float32(255)
    x = np.pad(x, ((0, 0), (0, coded_height - height), (0, coded_width - width)), mode="edge")
    return np.ascontiguousarray(x[np.newaxis], dtype="<f2")


def load_shifts(path, frames, gop):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("gop") != gop or metadata.get("frames", 0) < frames:
        raise ValueError("motion JSON GOP or frame coverage does not match the fixture")
    motion = metadata.get("motion", [])
    if len(motion) < frames:
        raise ValueError("motion JSON has too few frame records")
    shifts = []
    for index, row in enumerate(motion[:frames]):
        if row.get("frame") != index:
            raise ValueError(f"motion JSON is missing or reordering frame {index}")
        kx, ky = row.get("kx"), row.get("ky")
        if type(kx) is not int or type(ky) is not int or not (-128 <= kx <= 127 and -128 <= ky <= 127):
            raise ValueError(f"motion frame {index} is not signed int8 geometry")
        if index % gop == 0 and (kx or ky):
            raise ValueError(f"I-frame {index} contains nonzero geometry")
        shifts.append((index, kx, ky))
    return shifts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-json", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--gop", type=int, default=96)
    parser.add_argument("--coded-height", type=int, default=1088)
    parser.add_argument("--coded-width", type=int, default=1920)
    args = parser.parse_args()
    if args.frames <= 0 or args.gop <= 0 or args.coded_height % 8 or args.coded_width % 8:
        parser.error("frames/GOP must be positive and coded dimensions divisible by 8")
    shifts = load_shifts(args.motion_json, args.frames, args.gop)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("fixture output must be empty to avoid replacing existing data")
    hashes = []
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.base_rate or stream.average_rate)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "json", str(args.video)],
            check=True, capture_output=True, text=True)
        reference_fps = float(Fraction(json.loads(probe.stdout)["streams"][0]["r_frame_rate"]))
        if not np.isclose(fps, reference_fps, rtol=0, atol=1e-9):
            raise ValueError(f"PyAV base_rate {fps} differs from inference ffprobe r_frame_rate {reference_fps}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video has no valid frame rate")
        width, height = stream.codec_context.width, stream.codec_context.height
        count = 0
        for frame in container.decode(stream):
            if count == args.frames:
                break
            if frame.width != width or frame.height != height:
                raise ValueError("video changes frame dimensions")
            x = prepare_frame(frame, args.coded_height, args.coded_width)
            target = args.output / f"frame_{count}.fp16"
            x.tofile(target)
            hashes.append(hashlib.sha256(memoryview(x)).hexdigest())
            count += 1
        if count != args.frames:
            raise ValueError(f"video decoded {count} frames but {args.frames} requested")
    (args.output / "source_info.txt").write_text(
        f"width={width}\nheight={height}\nfps={fps:.12g}\n", encoding="utf-8")
    (args.output / "shifts.csv").write_text(
        "".join(f"{index},{kx},{ky}\n" for index, kx, ky in shifts), encoding="utf-8")
    report = {
        "video": str(args.video.resolve()), "motion_json": str(args.motion_json.resolve()),
        "frames": count, "visible_shape": [height, width],
        "coded_shape": [1, 3, args.coded_height, args.coded_width],
        "fps": fps, "dtype": "little-endian float16", "format": "NCHW YUV444",
        "normalize": "float32 sample/255 before float16 conversion",
        "chroma_upsampling": "nearest", "padding": "bottom/right replicate",
        "frame_sha256": hashes,
    }
    (args.output / "fixture.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frame_sha256"}, indent=2))


if __name__ == "__main__":
    main()
