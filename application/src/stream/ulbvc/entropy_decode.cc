#include <ulbvc/application/stream/ulbvc_entropy_decode.h>
#include <ulbvc/codec/detail/stage/constants.h>
#include <ulbvc/codec/ulbvc_entropy.h>
#include <ulbvc/codec/tensor_utils.h>
#include <ulbvc/core/status.h>

namespace ulbvc::codec {

DecodedEntropyFrame DecodeUlbvcEntropyFrame(ulbvc::UlbvcOfficialEntropyDecoder* decoder,
                                           const ulbvc::ModelRecord& decoder_record, int frame_index,
                                           UlbvcFrameType frame_type, int q_index,
                                           std::vector<uint8_t> payload, ulbvc::Profiler* profiler) {
  ulbvc::Check(decoder != nullptr, "ULBVC entropy decoder is required");
  ulbvc::Check(profiler != nullptr, "ULBVC entropy profiler is required");
  ulbvc::Check(decoder_record.inputs.size() >= 3,
              "ULBVCDecoder must provide z_raw, y_raw_0, and y_raw_1 inputs");

  const std::vector<int64_t>& z_shape = decoder_record.inputs.at(0).shape;
  const std::vector<int64_t>& y_shape_0 = decoder_record.inputs.at(1).shape;
  const std::vector<int64_t>& y_shape_1 = decoder_record.inputs.at(2).shape;
  ulbvc::Check(z_shape.size() == 4 && y_shape_0.size() == 4 && y_shape_1.size() == 4,
              "ULBVC entropy tensors must be rank four");
  const int y_channels = static_cast<int>(y_shape_0.at(1) * 2);
  const int y_height = static_cast<int>(y_shape_0.at(2));
  const int y_width = static_cast<int>(y_shape_0.at(3));
  const int z_height = static_cast<int>(z_shape.at(2));
  const int z_width = static_cast<int>(z_shape.at(3));

  ulbvc::ScopedCpuTimer timer(profiler, "entropy.rans_decode");
  DecodedEntropyFrame decoded;
  decoded.frame_index = frame_index;
  decoded.frame_type = frame_type;
  decoded.q_index = q_index;
  decoder->SetStream(payload, profiler);
  const std::size_t z_symbol_count = ulbvc::TensorShape(z_shape).NumElements();
  std::vector<int8_t> z_symbols =
      decoder->DecodeZ(q_index, static_cast<int>(z_symbol_count / (z_height * z_width)), z_height,
                       z_width, profiler);
  std::vector<uint8_t> scales_0;
  std::vector<uint8_t> scales_1;
  {
    ulbvc::ScopedCpuTimer stage_timer(profiler, "entropy.z.int8_to_fp16");
    FloatToFp16TensorFromInt8(z_symbols, ulbvc::TensorShape(z_shape), &decoded.z_raw);
  }
  {
    ulbvc::ScopedCpuTimer stage_timer(profiler, "entropy.scale_index_expand");
    BuildUlbvcScaleIndexesFromZRaw(decoded.z_raw, y_channels, y_height, y_width, kUlbvcChannelRepeat,
                                  &scales_0, &scales_1);
  }
  std::vector<int8_t> y_symbols_0 = decoder->DecodeY(scales_0, false, profiler);
  std::vector<int8_t> y_symbols_1 = decoder->DecodeY(scales_1, true, profiler);
  {
    ulbvc::ScopedCpuTimer stage_timer(profiler, "entropy.y0.int8_to_fp16");
    FloatToFp16TensorFromInt8(y_symbols_0, ulbvc::TensorShape(y_shape_0), &decoded.y_raw_0);
  }
  {
    ulbvc::ScopedCpuTimer stage_timer(profiler, "entropy.y1.int8_to_fp16");
    FloatToFp16TensorFromInt8(y_symbols_1, ulbvc::TensorShape(y_shape_1), &decoded.y_raw_1);
  }
  return decoded;
}

}  // namespace ulbvc::codec
