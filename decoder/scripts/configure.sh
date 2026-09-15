#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT}/build}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann}" \
  -DOpenCV_DIR="${OpenCV_DIR:-/opt/opencv/lib64/cmake/opencv4}" \
  -DMLVC_EXTERNAL_ROOT="${MLVC_EXTERNAL_ROOT:-${ROOT}/..}" \
  -DCMAKE_BUILD_TYPE=Release
