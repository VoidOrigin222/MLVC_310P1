#!/usr/bin/env python3
"""Evaluate reconstructed PNG frames against the actual model-input baseline."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def psnr(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((reference - reconstructed) ** 2, dtype=np.float64)
    return math.inf if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def main() -> None:
    parser = argparse.ArgumentParser()
    reference_source = parser.add_mutually_exclusive_group(required=True)
    reference_source.add_argument("--reference", type=Path)
    reference_source.add_argument("--reference-video", type=Path)
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument(
        "--reference-index-offset-after",
        nargs=2,
        type=int,
        metavar=("RECON_INDEX", "OFFSET"),
        help="For reconstructed frame indices >= RECON_INDEX, compare against reference index + OFFSET.",
    )
    args = parser.parse_args()

    reconstructed_paths = sorted(args.reconstructed.glob("im*.png"))
    if not reconstructed_paths:
        raise RuntimeError(f"No PNG files found in {args.reconstructed}")
    if args.reference is not None and not any(args.reference.glob("im*.png")):
        raise RuntimeError(f"No PNG files found in {args.reference}")
    args.output.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.skip_lpips:
        import lpips
        import torch

        if args.device == "npu":
            import torch_npu  # noqa: F401

        model = lpips.LPIPS(net="alex", version="0.1").eval().to(args.device)
    rows = []
    offset_start = offset = None
    if args.reference_index_offset_after is not None:
        offset_start, offset = args.reference_index_offset_after

    video_capture = None
    if args.reference_video is not None:
        import cv2

        video_capture = cv2.VideoCapture(str(args.reference_video))
        if not video_capture.isOpened():
            raise RuntimeError(f"Unable to open reference video: {args.reference_video}")

    for index, reconstructed_path in enumerate(reconstructed_paths):
        stem = reconstructed_path.stem
        if not stem.startswith("im") or not stem[2:].isdigit():
            raise RuntimeError(f"Unexpected reconstructed frame name: {reconstructed_path.name}")
        reconstructed_index = int(stem[2:])
        reference_index = reconstructed_index
        if offset_start is not None and reconstructed_index >= offset_start:
            reference_index += offset
        if video_capture is None:
            reference_path = args.reference / f"im{reference_index:05d}.png"
            if not reference_path.is_file():
                raise RuntimeError(f"Missing reference frame: {reference_path}")
            reference = read_rgb(reference_path)
            reference_file = reference_path.name
        else:
            success, bgr_frame = video_capture.read()
            if not success:
                raise RuntimeError(f"Reference video ended before {reconstructed_path.name}")
            reference = bgr_frame[:, :, ::-1].astype(np.float32)
            reference_file = f"video:{index}"
        reconstructed = read_rgb(reconstructed_path)
        if reference.shape != reconstructed.shape:
            raise RuntimeError(f"Shape mismatch for {reference_file}")
        row = {
            "frame_index": index,
            "reconstructed_file": reconstructed_path.name,
            "reference_file": reference_file,
            "psnr_rgb_db": psnr(reference, reconstructed),
        }
        if model is not None:
            with torch.inference_mode():
                reference_tensor = torch.from_numpy(reference.transpose(2, 0, 1)).unsqueeze(0)
                reconstructed_tensor = torch.from_numpy(reconstructed.transpose(2, 0, 1)).unsqueeze(0)
                reference_tensor = reference_tensor / 127.5 - 1.0
                reconstructed_tensor = reconstructed_tensor / 127.5 - 1.0
                reference_tensor = reference_tensor.to(args.device)
                reconstructed_tensor = reconstructed_tensor.to(args.device)
                row["lpips_alex"] = float(model(reference_tensor, reconstructed_tensor).item())
        rows.append(row)

    if video_capture is not None:
        video_capture.release()

    psnr_values = [row["psnr_rgb_db"] for row in rows]
    summary = {
        "frame_count": len(rows),
        "mean_psnr_rgb_db": float(np.mean(psnr_values)),
        "min_psnr_rgb_db": float(np.min(psnr_values)),
        "max_psnr_rgb_db": float(np.max(psnr_values)),
        "lpips_evaluated": not args.skip_lpips,
    }
    if model is not None:
        lpips_values = [row["lpips_alex"] for row in rows]
        summary.update({
            "mean_lpips_alex": float(np.mean(lpips_values)),
            "min_lpips_alex": float(np.min(lpips_values)),
            "max_lpips_alex": float(np.max(lpips_values)),
            "lpips_model": "pretrained AlexNet, lpips 0.1.4, version=0.1",
        })
    with (args.output / "per_frame_quality.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
