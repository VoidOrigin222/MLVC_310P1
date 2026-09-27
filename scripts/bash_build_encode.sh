#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/bash_build_encode.sh"
  echo "Environment: MLVC_CANN_HOME, OpenCV_DIR, MLVC_BUILD_DIR, BUILD_JOBS"
  exit 0
fi
CANN_HOME="${MLVC_CANN_HOME:-${CANN_HOME:-/usr/local/Ascend/cann-9.1.0}}"
BUILD_DIR="${MLVC_BUILD_DIR:-${ROOT}/build}"
BUILD_JOBS="${BUILD_JOBS:-4}"

if [[ ! -f "${CANN_HOME}/set_env.sh" && -f /usr/local/Ascend/cann/set_env.sh ]]; then
  CANN_HOME=/usr/local/Ascend/cann
fi
if [[ ! -f "${CANN_HOME}/set_env.sh" ]]; then
  echo "CANN set_env.sh not found under ${CANN_HOME}" >&2
  exit 1
fi
set +u
source "${CANN_HOME}/set_env.sh"
set -u

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_HOME}" \
    -DOpenCV_DIR="${OpenCV_DIR:-/opt/opencv/lib64/cmake/opencv4}" \
    -DCMAKE_BUILD_TYPE=Release
else
  cached_root="$(sed -n 's#^CMAKE_HOME_DIRECTORY:INTERNAL=##p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
  if [[ "${cached_root}" != "${ROOT}" ]]; then
    echo "CMake cache belongs to ${cached_root}, not ${ROOT}; choose another MLVC_BUILD_DIR or reconfigure explicitly." >&2
    exit 1
  fi
fi

cmake --build "${BUILD_DIR}" --target mlvc_encode --parallel "${BUILD_JOBS}"
