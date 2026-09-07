#!/usr/bin/env python3
"""Convert runtime_sidecar.npz into the C++ model sidecar format."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


DTYPE_IDS = {
    np.dtype("float16"): 0,
    np.dtype("float32"): 1,
    np.dtype("int32"): 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = np.load(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        out.write(b"ULBVC_SC")
        out.write(struct.pack("<II", 1, len(arrays.files)))
        for name in sorted(arrays.files):
            array = np.ascontiguousarray(arrays[name])
            dtype = array.dtype
            if dtype not in DTYPE_IDS:
                raise RuntimeError(f"unsupported dtype for {name}: {dtype}")
            encoded_name = name.encode("utf-8")
            out.write(struct.pack("<I", len(encoded_name)))
            out.write(encoded_name)
            out.write(struct.pack("<II", DTYPE_IDS[dtype], array.ndim))
            out.write(struct.pack("<" + "q" * array.ndim, *array.shape))
            payload = array.tobytes(order="C")
            out.write(struct.pack("<Q", len(payload)))
            out.write(payload)


if __name__ == "__main__":
    main()
