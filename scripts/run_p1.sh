#!/usr/bin/env bash
set -eo pipefail

deployment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
role="${1:-}"
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash scripts/run_p1.sh encode|decode' >&2
  exit 2
fi
case "${role}" in
  encode|decode) ;;
  *) echo 'Role must be encode or decode' >&2; exit 2 ;;
esac

cann_env="${CANN_HOME:-/usr/local/Ascend/cann}/set_env.sh"
if [[ -f "${cann_env}" ]]; then
  # shellcheck disable=SC1090
  source "${cann_env}"
fi
set -u
cd "${deployment_dir}"
exec "./build/mlvc_${role}" --config "configs/${role}.toml"
