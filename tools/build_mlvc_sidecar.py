#!/usr/bin/env python3
"""Build the compact C++ sidecar from MLVC PMF JSON assets."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_cdf_arrays(pmf: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(pmf["pmf_lengths"], dtype=np.int32)
    offsets = -np.asarray(pmf["pmf_offsets"], dtype=np.int32)
    table = np.asarray(pmf["pmf_table"], dtype=np.int64)
    cdf_lengths = lengths + 1
    width = int(cdf_lengths.max())
    cdf = np.zeros((len(lengths), width), dtype=np.int32)
    cursor = 0
    for row, length in enumerate(lengths):
        pmf_row = table[cursor : cursor + int(length)]
        values = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(pmf_row)))
        if values[-1] != 65536:
            raise ValueError(f"PMF row {row} does not sum to 65536: {values[-1]}")
        cdf[row, : values.size] = values.astype(np.int32)
        cursor += int(length)
    if cursor != len(table):
        raise ValueError("PMF table length does not match PMF lengths")
    return cdf, cdf_lengths, offsets


def write_array(stream, name: str, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    dtype_id = {np.dtype("float16"): 0, np.dtype("float32"): 1, np.dtype("int32"): 2}[array.dtype]
    name_bytes = name.encode("utf-8")
    stream.write(struct.pack("<I", len(name_bytes)))
    stream.write(name_bytes)
    stream.write(struct.pack("<II", dtype_id, array.ndim))
    stream.write(struct.pack("<" + "q" * array.ndim, *array.shape))
    payload = array.tobytes(order="C")
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gaussian_cdf, gaussian_lengths, gaussian_offsets = make_cdf_arrays(
        load_json(args.model_dir / "gaussian_pmf.json")
    )
    z_cdf, z_lengths, z_offsets = make_cdf_arrays(load_json(args.model_dir / "bit_estimator_pmf.json"))
    arrays = {
        "force_zero_thres": np.asarray([0.12], dtype=np.float32),
        "p_gaussian_cdf": gaussian_cdf,
        "p_gaussian_cdf_length": gaussian_lengths,
        "p_gaussian_offset": gaussian_offsets,
        "p_qp_offsets": np.asarray([0, 8, 4], dtype=np.int32),
        "p_z_cdf": z_cdf,
        "p_z_cdf_length": z_lengths,
        "p_z_channel": np.asarray([48], dtype=np.int32),
        "p_z_offset": z_offsets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"ULBVC_SC")
        stream.write(struct.pack("<II", 1, len(arrays)))
        for name in sorted(arrays):
            write_array(stream, name, arrays[name])


if __name__ == "__main__":
    main()
