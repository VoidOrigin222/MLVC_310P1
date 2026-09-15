#!/usr/bin/env bash
set -eo pipefail

deployment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
role="${1:-}"
resolution="${2:-720p}"
case "${role}" in
  encode|decode) ;;
  *) echo 'Usage: bash scripts/run_p1.sh encode|decode [720p|1080p]' >&2; exit 2 ;;
esac
case "${resolution}" in
  720p|1080p) ;;
  *) echo 'Resolution must be 720p or 1080p' >&2; exit 2 ;;
esac

source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u
cd "${deployment_dir}"
exec "./build/mlvc_${role}_v3" --config "configs/p1/${role}r_${resolution}.toml"
