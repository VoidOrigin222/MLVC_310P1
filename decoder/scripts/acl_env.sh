#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MLVC_ACL_REPO_ROOT="${repo_root}"
export CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${CANN_HOME}}"

if [ -f "${CANN_HOME}/set_env.sh" ]; then
  # shellcheck disable=SC1090
  set +u
  source "${CANN_HOME}/set_env.sh"
  set -u
fi

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/dcblock-acl/bin/python}"
if [ -x "$(dirname "${PYTHON_BIN}")/python" ]; then
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
fi

custom_opp_envs=(
  "${MLVC_ACL_REPO_ROOT}/output/custom_opp/wsiluchunkadd/vendors/mlvc/bin/set_env.bash"
  "${MLVC_ACL_REPO_ROOT}/output/custom_opp/mlvc_prior_ops/vendors/mlvc/bin/set_env.bash"
)
loaded_custom_opp_envs=()
for custom_opp_env in "${custom_opp_envs[@]}"; do
  if [ -f "${custom_opp_env}" ]; then
    # shellcheck disable=SC1090
    set +u
    source "${custom_opp_env}"
    set -u
    loaded_custom_opp_envs+=("${custom_opp_env}")
  fi
done

prior_opapi_lib="${MLVC_ACL_REPO_ROOT}/output/custom_opp/mlvc_prior_ops/vendors/mlvc/op_api/lib/libcust_opapi.so"
if [ -f "${prior_opapi_lib}" ]; then
  export MLVC_PRIOR_OPAPI_LIB="${prior_opapi_lib}"
fi

echo "MLVC_ACL_REPO_ROOT=${MLVC_ACL_REPO_ROOT}"
echo "CANN_HOME=${CANN_HOME}"
echo "PYTHON_BIN=${PYTHON_BIN}"
if [ "${#loaded_custom_opp_envs[@]}" -gt 0 ]; then
  printf 'custom_opp_envs=%s\n' "${loaded_custom_opp_envs[*]}"
else
  echo "custom_opp_envs=not-installed"
fi
echo "MLVC_PRIOR_OPAPI_LIB=${MLVC_PRIOR_OPAPI_LIB:-not-installed}"
