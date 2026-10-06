#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MLVC_ACL_REPO_ROOT="${MLVC_ACL_REPO_ROOT:-${ROOT}}"
export MLVC_BUILD_DIR="${MLVC_BUILD_DIR:-${ROOT}/camera_1080p_aipp/build}"
CONFIG="${1:-${ROOT}/configs/encoder.toml}"
CONFIG="$(realpath "${CONFIG}")"
cd "${ROOT}"
exec bash "${ROOT}/scripts/bash_run_encode.sh" "${CONFIG}"
