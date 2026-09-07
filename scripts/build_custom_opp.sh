#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${repo_root}/scripts/acl_env.sh"

build_and_install() {
  local op_name="$1"
  local project_dir="${repo_root}/custom_ops/${op_name}"
  local install_root="${repo_root}/output/custom_opp/${op_name}"

  if [ ! -d "${project_dir}" ]; then
    echo "custom OPP project not found: ${project_dir}" >&2
    exit 1
  fi

  (
    cd "${project_dir}"
    bash build.sh
  )

  mkdir -p "${install_root}"
  local run_pkg
  run_pkg="$(find "${project_dir}/build_out" -maxdepth 1 -type f -name 'custom_opp_*.run' | sort | tail -1)"
  if [ -z "${run_pkg}" ]; then
    echo "custom OPP run package not found under ${project_dir}/build_out" >&2
    exit 1
  fi

  "${run_pkg}" --quiet --install-path="${install_root}"
  echo "Installed ${op_name} custom OPP to ${install_root}"
  echo "Source environment with: source ${install_root}/vendors/mlvc/bin/set_env.bash"
}

build_and_install wsiluchunkadd
build_and_install mlvc_prior_ops
