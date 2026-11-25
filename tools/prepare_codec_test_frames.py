#!/usr/bin/env python3
"""Prepare a finite 1080P FP16 YUV444 sequence for codec/detector integration tests."""

import argparse
import os
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("image", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--frames", type=int, default=30)
parser.add_argument("--alternate-blank", action="store_true",
                    help="alternate the source image with a blank frame to test temporal association")
args = parser.parse_args()
if args.frames < 1 or (args.alternate_blank and args.frames < 2):
    parser.error("frames must be positive; alternating mode needs at least two")
args.output.mkdir(parents=True, exist_ok=True)
rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255
if rgb.shape != (1080, 1920, 3):
    raise ValueError("test source must be 1920x1080")
y = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
cb = 0.5 * (rgb[:, :, 2] - y) / (1 - 0.0722) + 0.5
cr = 0.5 * (rgb[:, :, 0] - y) / (1 - 0.2126) + 0.5
tensor = np.pad(np.stack([y, cb, cr]), ((0, 0), (0, 8), (0, 0)), mode="edge").astype(np.float16)
first = args.output / "frame_0.fp16"
if first.exists():
    raise FileExistsError(first)
tensor.tofile(first)
if args.alternate_blank:
    blank = np.full_like(tensor, 0.5)
    blank[0] = 114 / 255.0
    blank.tofile(args.output / "frame_1.fp16")
for index in range(2 if args.alternate_blank else 1, args.frames):
    source = args.output / "frame_1.fp16" if args.alternate_blank and index % 2 else first
    os.link(source, args.output / f"frame_{index}.fp16")
(args.output / "source_info.txt").write_text("width=1920\nheight=1080\nfps=30\n")
with tarfile.open(args.output.with_suffix(".tar.gz"), "w:gz") as archive:
    archive.add(args.output, arcname="frames")
print(args.output.with_suffix(".tar.gz"))
