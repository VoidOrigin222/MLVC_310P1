#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/bash_run.sh [decoder-config.toml]"
  echo "Environment: MLVC_CANN_HOME, MLVC_BUILD_DIR, MLVC_VECTOR_OPAPI_LIB, MLVC_FFMPEG_BIN"
  exit 0
fi
if [[ -f "${SCRIPT_DIR}/../CMakeLists.txt" ]]; then
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/source/CMakeLists.txt" ]]; then
  ROOT="${SCRIPT_DIR}/source"
else
  echo "Cannot locate the MLVC source tree next to ${SCRIPT_DIR}" >&2
  exit 1
fi
CANN_HOME="${MLVC_CANN_HOME:-${CANN_HOME:-/usr/local/Ascend/cann-9.1.0}}"
DEFAULT_BUILD_DIR="${ROOT}/build"
if [[ -f "${ROOT}/../build/CMakeCache.txt" &&
      -f "${ROOT}/../build/mlvc_decode" &&
      "$(sed -n 's#^CMAKE_HOME_DIRECTORY:INTERNAL=##p' "${ROOT}/../build/CMakeCache.txt" | head -n 1)" == "${ROOT}" ]]; then
  DEFAULT_BUILD_DIR="${ROOT}/../build"
fi
BUILD_DIR="${MLVC_BUILD_DIR:-${DEFAULT_BUILD_DIR}}"
DEFAULT_CONFIG="${ROOT}/configs/decoder.toml"
CONFIG="${1:-${DEFAULT_CONFIG}}"
VECTOR_OPAPI_LIB="${MLVC_VECTOR_OPAPI_LIB:-}"
FFMPEG_BIN="${MLVC_FFMPEG_BIN:-}"

if [[ ! -f "${CANN_HOME}/set_env.sh" && -f /usr/local/Ascend/cann/set_env.sh ]]; then
  CANN_HOME=/usr/local/Ascend/cann
fi
if [[ ! -f "${CANN_HOME}/set_env.sh" ]]; then
  echo "CANN set_env.sh not found under ${CANN_HOME}" >&2
  exit 1
fi
if [[ ! -x "${BUILD_DIR}/mlvc_decode" ]]; then
  echo "Decoder not built: ${BUILD_DIR}/mlvc_decode (run scripts/bash_build.sh first)" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Decoder config not found: ${CONFIG}" >&2
  exit 1
fi
# shellcheck disable=SC1090
set +u
source "${CANN_HOME}/set_env.sh"
set -u
if [[ -n "${VECTOR_OPAPI_LIB}" ]]; then
  if [[ ! -f "${VECTOR_OPAPI_LIB}" ]]; then
    echo "Vectorized operator library not found: ${VECTOR_OPAPI_LIB}" >&2
    exit 1
  fi
  export MLVC_PRIOR_OPAPI_LIB="${VECTOR_OPAPI_LIB}"
  echo "Using vectorized NV12 operator: ${MLVC_PRIOR_OPAPI_LIB}"
fi
if [[ -n "${FFMPEG_BIN}" ]]; then
  if [[ ! -x "${FFMPEG_BIN}/ffmpeg" ]]; then
    echo "FFmpeg runtime not found: ${FFMPEG_BIN}/ffmpeg" >&2
    exit 1
  fi
  export PATH="${FFMPEG_BIN}:${PATH}"
fi
exec "${BUILD_DIR}/mlvc_decode" --config "${CONFIG}"
