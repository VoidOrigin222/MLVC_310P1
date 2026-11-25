#ifndef ULBVC_CODEC_DETAIL_SCRATCH_H_
#define ULBVC_CODEC_DETAIL_SCRATCH_H_

#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/tensor.h>
#include <ulbvc/runtime/stage_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ulbvc::codec {

struct PriorScratch {
  std::vector<float> decoder_q_values;
  std::vector<float> scales;
  std::vector<float> means;
  std::vector<float> reconstructed;
  std::vector<float> part_scales;
  std::vector<int8_t> y_symbols;
  std::vector<int8_t> z_symbols;
  TensorData reconstructed_fp16;
  TensorData quantized_z;
  ulbvc::TensorHandle quantized_z_handle;
  TensorData output;
  ulbvc::TensorHandle reconstructed_fp16_handle;
  ulbvc::TensorHandle output_handle;
  ulbvc::AclBuffer y_symbols_acl;
  ulbvc::AclBuffer z_symbols_acl;
  ulbvc::AclBuffer part_scales_acl;
  std::vector<uint16_t> part_scales_fp16;
  ulbvc::TensorShape latent_shape;
  ulbvc::TensorShape z_shape;

  void ReserveImage(int height, int width) {
    constexpr int kImageChannels = 256;
    Reserve(kImageChannels, height, width);
  }

  void ReserveVideo(int height, int width) {
    constexpr int kVideoChannels = 128;
    Reserve(kVideoChannels, height, width);
  }

 private:
  void Reserve(int channels, int height, int width) {
    const std::size_t elements = static_cast<std::size_t>(channels * height * width);
    latent_shape = ulbvc::TensorShape({1, channels, height, width});
    const int z_height = (height + 3) / 4;
    const int z_width = (width + 3) / 4;
    z_shape = ulbvc::TensorShape({1, 128, z_height, z_width});
    decoder_q_values.reserve(elements);
    scales.reserve(elements);
    means.reserve(elements);
    reconstructed.reserve(elements);
    part_scales.reserve(elements);
    part_scales_fp16.reserve(elements);
    y_symbols.reserve(elements);
    z_symbols.reserve(elements);
    reconstructed_fp16.bytes.reserve(elements * ulbvc::ElementSize(ulbvc::DataType::kFloat16));
    quantized_z.bytes.reserve(elements * ulbvc::ElementSize(ulbvc::DataType::kFloat16));
    output.bytes.reserve(elements * ulbvc::ElementSize(ulbvc::DataType::kFloat16));
  }
};

struct CodecScratch {
  PriorScratch image_prior;
  PriorScratch video_prior;
  TensorData reference_feature;
  TensorData decoded_feature;
  TensorData x_hat_encode;
  TensorData x_hat_decode;

  void ReserveRoute(ulbvc::StageModelSet* models) {
    reference_feature = MakeTensorLike(
        OutputSpec(models->manifest().GetModel("p_reference_frame_adaptor"), "reference_feature"));
    decoded_feature = MakeTensorLike(
        OutputSpec(models->manifest().GetModel("p_synthesis_decoder"), "decoded_feature"));
    x_hat_encode = MakeTensorLike(OutputSpec(models->manifest().GetModel("i_decoder"), "x_hat"));
    x_hat_decode = MakeTensorLike(OutputSpec(models->manifest().GetModel("i_decoder"), "x_hat"));
  }
};

struct AsyncEntropyScratch {
  std::vector<int8_t> z_symbol_scratch;
  std::vector<int8_t> y_symbol_scratch;
  std::vector<float> scale_scratch;
  std::vector<int16_t> y_index_scratch;

  void Reserve(int channels, int height, int width) {
    const std::size_t elements = static_cast<std::size_t>(channels * height * width);
    z_symbol_scratch.reserve(elements);
    y_symbol_scratch.reserve(elements);
    scale_scratch.reserve(elements);
    y_index_scratch.reserve(elements);
  }
};

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_SCRATCH_H_
