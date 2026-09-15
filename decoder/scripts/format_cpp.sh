#!/usr/bin/env bash
set -euo pipefail

format_binary="${CLANG_FORMAT:-clang-format}"
command -v "${format_binary}" >/dev/null 2>&1 || {
  echo "clang-format is required; set CLANG_FORMAT to its path" >&2
  exit 1
}

mapfile -d '' source_files < <(
  find common mlvc application apps tools -type f \( -name '*.h' -o -name '*.cc' \) -print0
)
if ((${#source_files[@]} == 0)); then
  exit 0
fi

"${format_binary}" -i "${source_files[@]}"
