#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"
cmake -S "${ROOT}" -B "${ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH="${CANN_HOME}" \
  -DOpenCV_DIR="${ROOT}/sdk/opencv" \
  -DFFMPEG_INCLUDE_DIR="${ROOT}/sdk/ffmpeg/include" \
  -DFFMPEG_AVFORMAT_LIBRARY="${ROOT}/lib/libavformat.so.58" \
  -DFFMPEG_AVCODEC_LIBRARY="${ROOT}/lib/libavcodec.so.58" \
  -DFFMPEG_AVUTIL_LIBRARY="${ROOT}/lib/libavutil.so.56" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${ROOT}/lib"
cmake --build "${ROOT}/build" --parallel "${BUILD_JOBS:-4}"
