#!/usr/bin/env python3
"""Export the verified Python frame-reference path before feature_adaptor_p.

The output retains both feature and memory channels from feature_adaptor_i.
Existing Encoder/Decoder ONNX weights must prove the checkpoint has the same
feature_adaptor_p; this stage never applies that adaptor a second time.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

MODEL_CONFIG = {
    "type": "DMC-6.1sb", "activation": "LeakyReLU", "input_offset": -0.5,
    "feature_channels": 48, "spatial_prior_channels": 128,
    "memory_activation": "identity", "zero_init_residual": True,
    "chunk_mode": "gated", "ffn_gate_activation": "ReLU1", "chain_feature_adaptors": True,
    "recon_channels": 192, "hidden_channels": 192, "hyperprior_num_blocks": 2,
    "y_scale_repeat": 4, "z_channels": 48, "y_channels": 48,
    "hyperprior_variant": "mini", "feature_extractor_num_conv1_layers": 1,
    "feature_extractor_num_conv2_layers": 1,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReferenceFromFrame(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_adaptor_i = model.feature_adaptor_i
        self.input_offset = model.input_offset
        self.factor = model.pixel_shuffle_factor

    def forward(self, ref_frame):
        centered = ref_frame + self.input_offset if self.input_offset else ref_frame
        return self.feature_adaptor_i(F.pixel_unshuffle(centered, self.factor))


def check_existing_p_adaptors(paths, model):
    result = []
    for path in paths:
        graph = onnx.load(str(path))
        checked = []
        for suffix in ("feature_adaptor_p.conv.weight", "feature_adaptor_p.conv.bias"):
            tensors = [tensor for tensor in graph.graph.initializer if tensor.name.endswith(suffix)]
            if len(tensors) != 1:
                raise ValueError(f"{path}: expected exactly one initializer ending in {suffix}")
            array = onnx.numpy_helper.to_array(tensors[0])
            expected = model.state_dict()[suffix].detach().cpu().numpy().astype(np.float16)
            if array.dtype != np.float16 or array.shape != expected.shape or not np.array_equal(array, expected):
                raise ValueError(f"{path}: {suffix} is not exactly the checkpoint FP16 tensor")
            checked.append({"name": tensors[0].name, "shape": list(array.shape), "fp16_exact_match": True})
        result.append({"path": str(path.resolve()), "sha256": sha256(path), "initializers": checked})
    return result


def error_report(expected, actual, name):
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError(f"{name}: invalid ONNX output")
    error = np.abs(expected - actual)
    if not np.allclose(actual, expected, rtol=0.02, atol=0.02):
        raise ValueError(f"{name}: FP16 ONNX verification failed (max_abs={error.max()})")
    return {"max_abs": float(error.max()), "mean_abs": float(error.mean()), "rtol": 0.02, "atol": 0.02}


@torch.inference_mode()
def run(args):
    sys.path.insert(0, str(args.model_root.resolve()))
    from src.utils.model_factory import create_video_model
    from src.utils.stream_helper import get_state_dict

    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = create_video_model(MODEL_CONFIG)
    model.load_state_dict(get_state_dict(str(args.checkpoint)), strict=True)
    model.eval().to(args.device).half()
    if model.pixel_shuffle_factor != 8 or not model.chain_feature_adaptors or model.input_offset != -0.5:
        raise ValueError("model does not expose the expected chained feature/memory reference path")
    provenance = check_existing_p_adaptors([args.encoder_onnx, args.decoder_onnx], model)
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("export output already exists; choose a fresh validation directory")
    adaptor = ReferenceFromFrame(model).eval()
    shape = (1, 3, args.height, args.width)
    example = torch.full(shape, 0.5, dtype=torch.float16, device=args.device)
    torch.onnx.export(adaptor, (example,), str(args.output),
                      input_names=["ref_frame"], output_names=["ref_feature"],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.output), sess_options=session_options,
                                   providers=["CPUExecutionProvider"])
    cases = ["gop_constant_0_5", "reset_random_reconstructed_range"]
    verification = []
    for case in cases:
        ref_frame = example if case == cases[0] else torch.rand(shape, device=args.device).half()
        raw = adaptor(ref_frame)
        expected_shape = (1, 96, args.height // 8, args.width // 8)
        if tuple(raw.shape) != expected_shape:
            raise ValueError(f"unexpected ref_feature shape {tuple(raw.shape)}")
        raw_feature, raw_memory = raw.chunk(2, dim=1)
        expected_feature, expected_memory = model.apply_feature_adaptor(
            {"ref_frame": ref_frame, "ref_feature": None})
        torch.testing.assert_close(model.feature_adaptor_p(raw_feature), expected_feature, rtol=0, atol=0)
        torch.testing.assert_close(raw_memory, expected_memory, rtol=0, atol=0)
        input_np = ref_frame.cpu().numpy()
        actual = session.run(["ref_feature"], {"ref_frame": input_np})[0]
        metrics = error_report(raw.cpu().numpy(), actual, case)
        actual_feature = torch.from_numpy(actual[:, :48].copy()).to(args.device)
        chained_metrics = error_report(expected_feature.cpu().numpy(),
                                       model.feature_adaptor_p(actual_feature).cpu().numpy(),
                                       case + "_then_p_adaptor")
        input_np.astype("<f2").tofile(output_dir / (case + "_input.fp16"))
        raw.cpu().numpy().astype("<f2").tofile(output_dir / (case + "_expected.fp16"))
        verification.append({"case": case, "python_split_then_p_exact": True,
                             "onnx_pre_p": metrics, "onnx_then_p": chained_metrics,
                             "memory_abs_max": float(raw_memory.abs().max())})
    report = {"stage": "MLVCReferenceFromFrame", "onnx": str(args.output.resolve()),
              "onnx_sha256": sha256(args.output), "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": sha256(args.checkpoint), "model_config": MODEL_CONFIG,
              "input": {"name": "ref_frame", "shape": list(shape), "dtype": "float16", "range": [0, 1]},
              "output": {"name": "ref_feature", "shape": list(expected_shape), "dtype": "float16",
                         "channels": "feature[0:48] and memory[48:96] before p-adaptor"},
              "existing_p_adaptors": provenance, "verification": verification}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True, help="Python video directory containing src/")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-onnx", type=Path, required=True)
    parser.add_argument("--decoder-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8 or args.threads <= 0:
        parser.error("positive dimensions divisible by 8 and positive threads are required")
    run(args)


if __name__ == "__main__":
    main()
