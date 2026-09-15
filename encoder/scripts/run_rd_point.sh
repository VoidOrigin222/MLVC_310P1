#!/usr/bin/env bash
# Run one fixed-QP rate-distortion point from a named quality configuration.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <614lab_1080p|614lab_720p|60s_1080p|60s_720p> <qp> [--full|--decode-only]" >&2
  exit 2
fi

case_name="$1"
qp="$2"
mode="${3:-}"
case "$case_name" in
  614lab_1080p|614lab_720p|60s_1080p|60s_720p) ;;
  *) echo "unknown case: $case_name" >&2; exit 2 ;;
esac
if [[ ! "$qp" =~ ^([0-9]|[1-5][0-9]|6[0-3])$ ]]; then
  echo "qp must be an integer in [0, 63]" >&2
  exit 2
fi
if [[ -n "$mode" && "$mode" != "--full" && "$mode" != "--decode-only" ]]; then
  echo "third argument must be --full or --decode-only" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="$project_root/../rd_target_qp_20260902/$case_name/qp$qp"
encode_template="$project_root/configs/quality/encode_$case_name.toml"
decode_template="$project_root/configs/quality/decode_$case_name.toml"
encode_config="$(mktemp)"
decode_config="$(mktemp)"
trap 'rm -f "$encode_config" "$decode_config"' EXIT
mkdir -p "$result_dir"

encode_output="../rd_target_qp_20260902/$case_name/qp$qp/bitstream.mlvc"
sed -e "s|^output = \".*\"|output = \"$encode_output\"|" \
    -e "s|^qp = [0-9][0-9]*|qp = $qp|" "$encode_template" > "$encode_config"

cd "$project_root"
source scripts/env.sh
if [[ "$mode" != "--decode-only" ]]; then
  ./build/mlvc_encode --config "$encode_config" | tee "$result_dir/encode.log"
fi

if [[ "$mode" != "--full" && "$mode" != "--decode-only" ]]; then
  exit 0
fi

if [[ ! -s "$result_dir/bitstream.mlvc" ]]; then
  echo "missing bitstream for decode-only run: $result_dir/bitstream.mlvc" >&2
  exit 1
fi

decode_input="../rd_target_qp_20260902/$case_name/qp$qp/bitstream.mlvc"
decode_output="../rd_target_qp_20260902/$case_name/qp$qp/recon_png"
sed -e "s|^input = \".*\"|input = \"$decode_input\"|" \
    -e "s|^output = \".*\"|output = \"$decode_output\"|" "$decode_template" > "$decode_config"
./build/mlvc_decode --config "$decode_config" | tee "$result_dir/decode.log"

case "$case_name" in
  614lab_1080p) reference_video="../quality_inputs_20260902/614lab_1920x1080_sdr_bt709_lossless.mp4" ;;
  614lab_720p) reference_video="../quality_inputs_20260902/614lab_1280x720_sdr_bt709_lossless.mp4" ;;
  60s_1080p) reference_video="../quality_inputs_20260902/60s_1920x1080_30fps_lossless.mp4" ;;
  60s_720p) reference_video="../../ulbvc_acl_cpp/test_video/60s.mp4" ;;
esac
/root/miniconda3/envs/vllm-omni/bin/python scripts/evaluate_recon_quality.py \
    --reference-video "$reference_video" --reconstructed "$result_dir/recon_png" \
    --output "$result_dir/quality" --device npu | tee "$result_dir/quality.log"
ffmpeg -y -hide_banner -loglevel error -framerate 30 -start_number 1 \
    -i "$result_dir/recon_png/im%05d.png" -an -c:v libx264 -preset medium -crf 18 \
    -pix_fmt yuv420p -movflags +faststart "$result_dir/recon.mp4"
