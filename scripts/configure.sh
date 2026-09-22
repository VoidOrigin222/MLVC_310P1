#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT}/build}"

# A deployment directory can be renamed between releases.  Reusing a CMake
# cache from the old absolute source path makes CMake generate files for the
# wrong tree and can silently leave an old binary in use.  Preserve that
# build for rollback, then configure a clean directory for the current tree.
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  cached_root="$(sed -n 's#^CMAKE_HOME_DIRECTORY:INTERNAL=##p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
  if [[ -n "${cached_root}" && "${cached_root}" != "${ROOT}" ]]; then
    stale_dir="${BUILD_DIR}.stale-$(date +%Y%m%d-%H%M%S)"
    mv "${BUILD_DIR}" "${stale_dir}"
    echo "Moved stale CMake build (${cached_root}) to ${stale_dir}" >&2
  fi
fi

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann}" \
  -DOpenCV_DIR="${OpenCV_DIR:-/opt/opencv/lib64/cmake/opencv4}" \
  -DCMAKE_BUILD_TYPE=Release
