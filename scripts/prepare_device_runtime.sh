#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="${1:?existing role-specific runtime bundle is required}"
HEADERS="${2:?SDK header archive is required}"
mkdir -p "${ROOT}/lib" "${ROOT}/runtime-bin" "${ROOT}/sdk" "${ROOT}/logs"
cp -a "${BUNDLE}/lib/." "${ROOT}/lib/"
for program in ffmpeg ffprobe; do
  if [[ -f "${BUNDLE}/bin/${program}" ]]; then
    cp -a "${BUNDLE}/bin/${program}" "${ROOT}/runtime-bin/"
  fi
done
tar -mxzf "${HEADERS}" -C "${ROOT}/sdk"
cat > "${ROOT}/sdk/opencv/OpenCVConfig.cmake" <<'CMAKE'
set(OpenCV_FOUND TRUE)
set(OpenCV_VERSION "4.10.0")
set(OpenCV_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")
set(OpenCV_LIBS
  "${CMAKE_CURRENT_LIST_DIR}/../../lib/libopencv_core.so.410"
  "${CMAKE_CURRENT_LIST_DIR}/../../lib/libopencv_imgproc.so.410"
  "${CMAKE_CURRENT_LIST_DIR}/../../lib/libopencv_imgcodecs.so.410"
  "${CMAKE_CURRENT_LIST_DIR}/../../lib/libopencv_videoio.so.410")
CMAKE
