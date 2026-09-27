#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/bash_run_encode.sh [encoder-config.toml]"
  exit 0
fi
CANN_HOME="${MLVC_CANN_HOME:-${CANN_HOME:-/usr/local/Ascend/cann-9.1.0}}"
BUILD_DIR="${MLVC_BUILD_DIR:-${ROOT}/build}"
DEFAULT_CONFIG="${ROOT}/configs/encoder.toml"
CONFIG="${1:-${DEFAULT_CONFIG}}"

if [[ ! -f "${CANN_HOME}/set_env.sh" && -f /usr/local/Ascend/cann/set_env.sh ]]; then
  CANN_HOME=/usr/local/Ascend/cann
fi
if [[ ! -f "${CANN_HOME}/set_env.sh" ]]; then
  echo "CANN set_env.sh not found under ${CANN_HOME}" >&2
  exit 1
fi
if [[ ! -x "${BUILD_DIR}/mlvc_encode" ]]; then
  echo "Encoder not built: ${BUILD_DIR}/mlvc_encode (run scripts/bash_build_encode.sh first)" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Encoder config not found: ${CONFIG}" >&2
  exit 1
fi

set +u
source "${CANN_HOME}/set_env.sh"
set -u
exec "${BUILD_DIR}/mlvc_encode" --config "${CONFIG}"
