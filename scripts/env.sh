#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CANN_HOME="${ULBVC_CANN_HOME:-${CANN_HOME:-/usr/local/Ascend/cann-9.1.0}}"
source "${ROOT}/scripts/acl_env.sh"
export LD_LIBRARY_PATH="${ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ROOT}/runtime-bin:${PATH}"
export ULBVC_CODEC_RUNTIME_ENV_BOOTSTRAPPED=1
