#!/usr/bin/env python3
"""Export static YOLO11 detection ONNX models and verify their raw outputs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile


PROFILES = {
    "1080p": {"image_hw": (1080, 1920), "input_hw": (1088, 1920)},
    "720p": {"image_hw": (720, 1280), "input_hw": (736, 1280)},
}
REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = REPO / "models/detection/yolo11s"
    parser.add_argument("--weights", type=Path, default=default_root / "yolo11s.pt")
    parser.add_argument("--output", type=Path, default=default_root)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--image", type=Path, required=True,
                        help="1920x1080 image used for the numerical comparison")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--opset", type=int, choices=(13, 17), default=17)
    args = parser.parse_args()
    weights, image_path = args.weights.resolve(), args.image.resolve()
    if not weights.is_file() or not image_path.is_file():
        parser.error("weights and validation image must exist locally")
    if args.threads < 1:
        parser.error("threads must be positive")
    output = args.output.resolve()
    if args.precision == "fp16":
        output = output / f"fp16_opset{args.opset}"
    output = output / args.profile
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_fp16_opset{args.opset}" if args.precision == "fp16" else ""
    target = output / f"yolo11s_{args.profile}{suffix}.onnx"
    if target.exists():
        parser.error(f"output already exists: {target}; use another --output directory")

    settings = REPO / "artifacts/yolo11-export-settings"
    settings.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import numpy as np
    import onnx
    import onnxruntime as ort
    from PIL import Image
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    image_h, image_w = PROFILES[args.profile]["image_hw"]
    input_h, input_w = PROFILES[args.profile]["input_hw"]
    pad_top, pad_left = (input_h - image_h) // 2, (input_w - image_w) // 2
    with Image.open(image_path) as original:
        if original.size != (1920, 1080):
            parser.error("validation image must be 1920x1080")
        pixels = np.asarray(original.convert("RGB").resize(
            (image_w, image_h), Image.Resampling.BILINEAR))
    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    padded[pad_top:pad_top + image_h, pad_left:pad_left + image_w] = pixels
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    if args.precision == "fp16":
        tensor = tensor.astype(np.float16)
    del pixels, padded

    with tempfile.TemporaryDirectory(prefix="ulbvc-yolo11-", dir=output) as staging:
        checkpoint = Path(staging) / "yolo11s.pt"
        shutil.copyfile(weights, checkpoint)
        model = YOLO(str(checkpoint), task="detect")
        if int(model.model.stride.max()) != 32 or len(model.names) != 80:
            raise ValueError("expected the official stride-32, 80-class YOLO11 detector")
        labels = [model.names[i] for i in range(len(model.names))]
        reference_model = model.model.float().eval().fuse(verbose=False)
        if args.precision == "fp16":
            # The stock exporter disables half precision on CPU. Export the
            # actual half-precision network with fixed shapes instead.
            from ultralytics.nn.modules import Detect
            import onnxslim
            reference_model.half()
            for module in reference_model.modules():
                if isinstance(module, Detect):
                    module.export = True
                    module.format = "onnx"
                    module.dynamic = False
            exported = Path(staging) / "yolo11s.onnx"
            sample = torch.from_numpy(tensor)
            with torch.inference_mode():
                reference_model(sample)
                torch.onnx.export(
                    reference_model, sample, str(exported), opset_version=args.opset,
                    input_names=["images"], output_names=["output0"],
                    dynamic_axes=None, do_constant_folding=True, dynamo=False,
                )
            graph = onnxslim.slim(onnx.load(str(exported)))
            names = {"names": str(dict(enumerate(labels))), "task": "detect",
                     "imgsz": str([input_h, input_w]), "stride": "32", "batch": "1"}
            onnx.helper.set_model_props(graph, names)
            onnx.save(graph, str(exported))
            del graph, sample
        else:
            exported = Path(model.export(
                format="onnx", imgsz=(input_h, input_w), batch=1,
                dynamic=False, opset=args.opset, simplify=True, nms=False,
                half=False, device="cpu", verbose=False,
            ))
        with torch.inference_mode():
            reference_input = torch.from_numpy(tensor)
            reference = reference_model(reference_input)
            if isinstance(reference, tuple):
                reference = reference[0]
            reference = reference.detach().cpu().numpy().copy()
        del model, reference_model
        gc.collect()

        onnx.checker.check_model(str(exported), full_check=True)
        graph = onnx.load(str(exported))
        if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
            raise ValueError("the exported graph must expose raw predictions without NMS")
        if next(item.version for item in graph.opset_import if item.domain == "") != args.opset:
            raise ValueError("unexpected ONNX opset")
        ir_version = graph.ir_version
        del graph
        gc.collect()

        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(exported), options, providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        anchors = sum((input_h // stride) * (input_w // stride) for stride in (8, 16, 32))
        if len(inputs) != 1 or inputs[0].shape != [1, 3, input_h, input_w]:
            raise ValueError("unexpected or dynamic model input")
        if len(outputs) != 1 or outputs[0].shape != [1, 84, anchors]:
            raise ValueError("unexpected or dynamic model output")
        expected_type = "tensor(float16)" if args.precision == "fp16" else "tensor(float)"
        if inputs[0].type != expected_type or outputs[0].type != expected_type:
            raise ValueError("model I/O precision does not match the requested precision")
        actual = session.run(None, {inputs[0].name: tensor})[0]
        if not np.isfinite(actual).all():
            raise ValueError("ONNX inference returned non-finite values")
        if args.precision == "fp16":
            np.testing.assert_allclose(actual[:, :4], reference[:, :4], rtol=0.02, atol=4.0)
            np.testing.assert_allclose(actual[:, 4:], reference[:, 4:], rtol=0.05, atol=0.01)
        else:
            np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=5e-3)
        errors = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
        validation = {
            "onnx_checker": "passed", "onnxruntime_cpu": "passed",
            "pytorch_comparison": "passed",
            "pytorch_reference_precision": args.precision.upper(),
            "comparison_tolerance": ({"boxes_rtol": 0.02, "boxes_atol_px": 4.0,
                                      "scores_rtol": 0.05, "scores_atol": 0.01}
                                     if args.precision == "fp16" else {"rtol": 1e-3, "atol": 5e-3}),
            "max_absolute_error": float(errors.max()),
            "mean_absolute_error": float(errors.mean()),
            "box_max_absolute_error_pixels": float(errors[:, :4].max()),
            "class_score_max_absolute_error": float(errors[:, 4:].max()),
            "validation_image_sha256": sha256(image_path),
            "accuracy_dataset_evaluation": "not performed",
            "ascend310p1_compilation_and_runtime": "not performed",
        }
        input_record = {"name": inputs[0].name, "shape": inputs[0].shape, "dtype": inputs[0].type}
        output_record = {"name": outputs[0].name, "shape": outputs[0].shape, "dtype": outputs[0].type}
        tensor.tofile(output / "validation_input.bin")
        actual.tofile(output / "validation_output.bin")
        del session, actual, reference, errors
        gc.collect()
        exported.rename(target)

    gain = image_w / 1920.0
    metadata = {
        "model": "YOLO11s", "task": "detect", "profile": args.profile,
        "weights_sha256": sha256(weights), "file": target.name,
        "sha256": sha256(target), "bytes": target.stat().st_size,
        "format": "ONNX", "opset": args.opset, "ir_version": ir_version,
        "precision": args.precision.upper(),
        "input": input_record, "output": output_record,
        "output_layout": "[batch, 4 xywh box coordinates + 80 class scores, candidates]",
        "postprocess": "class filtering, confidence threshold and NMS in application",
        "class_names": labels,
        "preprocess": {
            "color": "RGB", "layout": "NCHW", "scale": "pixel / 255.0",
            "source_wh": [1920, 1080], "resized_wh": [image_w, image_h],
            "input_wh": [input_w, input_h], "resize_gain": gain,
            "padding_ltrb": [pad_left, pad_top, input_w - image_w - pad_left,
                             input_h - image_h - pad_top], "padding_value": 114,
            "restore_center_xy": "((cx - pad_left) / resize_gain, (cy - pad_top) / resize_gain)",
            "restore_wh": "(width / resize_gain, height / resize_gain)",
            "clip_boxes_to_source": True,
        },
        "export_arguments": {
            "batch": 1, "dynamic": False, "nms": False, "simplify": True, "device": "cpu",
            "precision": args.precision, "opset": args.opset,
            "fp16_export_method": "native" if args.precision == "fp16" else None,
        },
        "versions": {name: importlib.metadata.version(name) for name in
                     ("ultralytics", "torch", "torchvision", "onnx", "onnxruntime", "onnxslim", "numpy")},
        "validation": validation,
    }
    write_json(output / "metadata.json", metadata)
    (output / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (output / (target.name + ".sha256")).write_text(
        metadata["sha256"] + "  " + target.name + "\n", encoding="ascii")
    print(json.dumps({"file": str(target), "input": input_record, "output": output_record,
                      "sha256": metadata["sha256"], "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
