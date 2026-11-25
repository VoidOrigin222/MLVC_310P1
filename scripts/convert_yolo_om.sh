#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"
MODEL="${1:?ONNX model is required}"
OUTPUT="${2:?OM output prefix is required}"
HEIGHT="${3:?input height is required}"
WIDTH="${4:?input width is required}"
mkdir -p "$(dirname "${OUTPUT}")"
exec atc --framework=5 --model="${MODEL}" --output="${OUTPUT}" \
  --soc_version=Ascend310P1 --input_format=NCHW \
  --input_shape="images:1,3,${HEIGHT},${WIDTH}" \
  --output_type=FP16 --precision_mode=allow_fp32_to_fp16 --log=info
