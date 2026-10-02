#!/usr/bin/env python3
"""Audit the exact feature/memory boundary before enabling an existing OM pair.

Dependencies: numpy and onnx. This inspects ONNX, not OM compiler provenance;
OM hashes in the report must additionally match the conversion bundle being
deployed. The runtime allowlist accepts only the reviewed 1080p FP16 pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def shift(value: np.ndarray, kx: int, ky: int) -> np.ndarray:
    h, w = value.shape[-2:]
    require(abs(kx) < w and abs(ky) < h, "shift exceeds spatial shape")
    result = value.copy()
    result[..., max(ky, 0):min(h + ky, h), max(kx, 0):min(w + kx, w)] = value[
        ..., max(-ky, 0):min(h - ky, h), max(-kx, 0):min(w - kx, w)]
    return result


def audit(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    model = onnx.load(path)
    graph = model.graph
    initializers = {x.name: numpy_helper.to_array(x) for x in graph.initializer}
    reference = next(x for x in graph.input if x.name == "ref_feature")
    shape = [d.dim_value for d in reference.type.tensor_type.shape.dim]
    require(shape == [1, 96, 136, 240], "only the 1080p FP16 feature grid is audited")
    require(reference.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16,
            "reference must use FP16")

    def consumers(name: str) -> list:
        return [node for node in graph.node if name in node.input]

    splits = consumers("ref_feature")
    require(len(splits) == 2 and all(x.op_type == "Slice" for x in splits),
            "unexpected reference consumers")
    halves = {}
    for node in splits:
        require(len(node.input) in (4, 5), "unexpected Slice interface")
        require(all(name in initializers for name in node.input[1:]),
                "reference Slice bounds must be constant")
        starts, ends, axes = [initializers[name].tolist() for name in node.input[1:4]]
        require(axes == [1] and starts in ([0], [48]) and ends == [starts[0] + 48],
                "reference must split channels into feature and memory")
        if len(node.input) == 5:
            require(initializers[node.input[4]].tolist() == [1], "unexpected Slice stride")
        halves[starts[0]] = node.output[0]
    require(set(halves) == {0, 48}, "reference halves overlap or are missing")
    adaptors = consumers(halves[0])
    require(len(adaptors) == 1 and adaptors[0].op_type == "Conv", "unexpected feature adaptor")
    adaptor = adaptors[0]
    require(adaptor.name == "/feature_adaptor_p/conv/Conv", "unknown adaptor boundary")
    attributes = {a.name: onnx.helper.get_attribute_value(a) for a in adaptor.attribute}
    for key, expected in {"kernel_shape": [1, 1], "pads": [0, 0, 0, 0],
                          "strides": [1, 1], "dilations": [1, 1], "group": 1}.items():
        require(attributes.get(key, expected) == expected, f"adaptor {key} is not pointwise")
    require(attributes.get("auto_pad", b"NOTSET") in (b"NOTSET", b"VALID"), "adaptor auto padding")
    weights, bias = [initializers[name] for name in adaptor.input[1:]]
    require(weights.shape == (48, 48, 1, 1) and bias.shape == (48,), "unexpected adaptor weights")
    feature_consumers = consumers(adaptor.output[0])
    require(len(feature_consumers) == 1 and feature_consumers[0].name.startswith("/feature_extractor/"),
            "adapted feature has unknown consumers")
    memory_consumers = consumers(halves[48])
    require(len(memory_consumers) == 1 and memory_consumers[0].name == "/decoder/Mul_3" and
            memory_consumers[0].op_type == "Mul", "memory has unknown consumers")
    result = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "reference_shape": shape, "feature_slice": [0, 48], "memory_slice": [48, 96],
              "adaptor": adaptor.name, "adaptor_attributes": {
                  k: v.decode() if isinstance(v, bytes) else v for k, v in attributes.items()},
              "memory_consumer": memory_consumers[0].name}
    return result, weights[:, :, 0, 0], bias


def equivalence(weights: np.ndarray, bias: np.ndarray) -> dict:
    rng = np.random.default_rng(20261002)
    feature = rng.normal(size=(1, 48, 7, 9)).astype(np.float16)
    memory = rng.normal(size=feature.shape).astype(np.float16)

    def adaptor(value: np.ndarray) -> np.ndarray:
        # Identical ordered per-position arithmetic on both sides of selection.
        return (np.einsum("oc,bchw->bohw", weights.astype(np.float32),
                          value.astype(np.float32), optimize=False) +
                bias.astype(np.float32)[None, :, None, None]).astype(np.float16)

    offsets = [(0, 0), (1, 0), (-1, 0), (0, 2), (0, -2), (2, -3), (-4, 3), (8, 6), (-8, -6)]
    for kx, ky in offsets:
        before = shift(np.concatenate((feature, memory), axis=1), kx, ky)
        adapted_before = adaptor(before[:, :48])
        adapted_after = shift(adaptor(feature), kx, ky)
        require(np.array_equal(adapted_before.view(np.uint16), adapted_after.view(np.uint16)),
                f"pointwise adaptor/shift did not commute at {kx},{ky}")
        require(np.array_equal(before[:, 48:], shift(memory, kx, ky)), "memory shift mismatch")
    return {"checked_offsets": offsets, "fp16_bitwise_equal": True,
            "note": "host per-position arithmetic proof; separately validate compiled OM execution"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", type=Path)
    parser.add_argument("decoder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    enc, ew, eb = audit(args.encoder)
    dec, dw, db = audit(args.decoder)
    require(np.array_equal(ew, dw) and np.array_equal(eb, db), "encoder/decoder adaptors differ")
    report = {"schema_version": 1, "status": "passed", "models": [enc, dec],
              "equivalence": equivalence(ew, eb),
              "reason": "Integer spatial selection with preserved boundaries commutes with the shared 1x1 affine adaptor; memory is the identity second half."}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
