#include <mlvc/io/fp16_yuv444_to_nv12.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "mlvc/core/status.h"

namespace mlvc::io {
namespace {

void ValidateInput(const codec::TensorData& input, const Nv12Layout& layout,
                   const std::vector<uint8_t>* output) {
  Check(output != nullptr, "NV12 output is required");
  Check(input.dtype == DataType::kFloat16, "NV12 conversion expects an FP16 tensor");
  Check(input.shape.rank() == 4 && input.shape.dim(0) == 1 && input.shape.dim(1) == 3,
        "NV12 conversion expects an NCHW three-plane tensor");
  Check(layout.width > 0 && layout.height > 0 && layout.width % 2 == 0 &&
            layout.height % 2 == 0,
        "NV12 visible dimensions must be positive and even");
  Check(layout.width_stride >= layout.width && layout.height_stride >= layout.height &&
            layout.width_stride % 2 == 0 && layout.height_stride % 2 == 0,
        "NV12 strides must cover the visible image and be even");
  Check(input.shape.dim(3) >= layout.width && input.shape.dim(2) >= layout.height,
        "FP16 input tensor is smaller than the visible NV12 image");
}

uint8_t NormalizedToByte(float value) {
  value = std::min(std::max(value, 0.0F), 1.0F);
  return static_cast<uint8_t>(std::lrint(value * 255.0F));
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
uint8x8_t NormalizedHalf8ToBytes(const uint16_t* input) {
  const float16x8_t half = vreinterpretq_f16_u16(vld1q_u16(input));
  const float32x4_t zero = vdupq_n_f32(0.0F);
  const float32x4_t one = vdupq_n_f32(1.0F);
  const float32x4_t scale = vdupq_n_f32(255.0F);
  const float32x4_t low = vmulq_f32(vminq_f32(vmaxq_f32(vcvt_f32_f16(vget_low_f16(half)), zero), one), scale);
  const float32x4_t high = vmulq_f32(vminq_f32(vmaxq_f32(vcvt_f32_f16(vget_high_f16(half)), zero), one), scale);
  return vmovn_u16(vcombine_u16(vmovn_u32(vcvtnq_u32_f32(low)),
                                vmovn_u32(vcvtnq_u32_f32(high))));
}

float32x4_t AverageHalf2x2(const uint16_t* row0, const uint16_t* row1) {
  const float16x8_t half0 = vreinterpretq_f16_u16(vld1q_u16(row0));
  const float16x8_t half1 = vreinterpretq_f16_u16(vld1q_u16(row1));
  const float32x4_t pairs0 =
      vpaddq_f32(vcvt_f32_f16(vget_low_f16(half0)), vcvt_f32_f16(vget_high_f16(half0)));
  const float32x4_t pairs1 =
      vpaddq_f32(vcvt_f32_f16(vget_low_f16(half1)), vcvt_f32_f16(vget_high_f16(half1)));
  return vmulq_n_f32(vaddq_f32(pairs0, pairs1), 0.25F);
}

uint16x4_t NormalizedFloat4ToU16(float32x4_t input) {
  const float32x4_t clamped =
      vminq_f32(vmaxq_f32(input, vdupq_n_f32(0.0F)), vdupq_n_f32(1.0F));
  return vmovn_u32(vcvtnq_u32_f32(vmulq_n_f32(clamped, 255.0F)));
}
#endif

}  // namespace

std::size_t Nv12BufferSize(const Nv12Layout& layout) {
  Check(layout.width_stride > 0 && layout.height_stride > 0 &&
            layout.width_stride % 2 == 0 && layout.height_stride % 2 == 0,
        "NV12 strides must be positive and even");
  const std::size_t stride = static_cast<std::size_t>(layout.width_stride);
  const std::size_t height = static_cast<std::size_t>(layout.height_stride);
  Check(height <= std::numeric_limits<std::size_t>::max() / stride,
        "NV12 buffer size overflow");
  const std::size_t y_bytes = stride * height;
  Check(y_bytes <= std::numeric_limits<std::size_t>::max() / 3,
        "NV12 buffer size overflow");
  return y_bytes + y_bytes / 2;
}

bool Fp16Yuv444ToNv12UsesNeon() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  return true;
#else
  return false;
#endif
}

void ConvertFp16Yuv444ToNv12Scalar(const codec::TensorData& input,
                                   const Nv12Layout& layout,
                                   std::vector<uint8_t>* output) {
  ValidateInput(input, layout, output);
  output->assign(Nv12BufferSize(layout), 0);

  const int input_height = static_cast<int>(input.shape.dim(2));
  const int input_width = static_cast<int>(input.shape.dim(3));
  const std::size_t plane = static_cast<std::size_t>(input_height) * input_width;
  const auto* source = reinterpret_cast<const uint16_t*>(input.bytes.data());
  for (int y = 0; y < layout.height; ++y) {
    for (int x = 0; x < layout.width; ++x) {
      const std::size_t input_offset = static_cast<std::size_t>(y) * input_width + x;
      const std::size_t output_offset = static_cast<std::size_t>(y) * layout.width_stride + x;
      (*output)[output_offset] = NormalizedToByte(codec::HalfBitsToFloat(source[input_offset]));
    }
  }

  const std::size_t uv_base =
      static_cast<std::size_t>(layout.width_stride) * layout.height_stride;
  for (int y = 0; y < layout.height; y += 2) {
    for (int x = 0; x < layout.width; x += 2) {
      float u_sum = 0.0F;
      float v_sum = 0.0F;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const std::size_t offset = static_cast<std::size_t>(y + dy) * input_width + x + dx;
          u_sum += codec::HalfBitsToFloat(source[plane + offset]);
          v_sum += codec::HalfBitsToFloat(source[2 * plane + offset]);
        }
      }
      const std::size_t uv_offset =
          uv_base + static_cast<std::size_t>(y / 2) * layout.width_stride + x;
      (*output)[uv_offset] = NormalizedToByte(u_sum * 0.25F);
      (*output)[uv_offset + 1] = NormalizedToByte(v_sum * 0.25F);
    }
  }
}

void ConvertFp16Yuv444ToNv12(const codec::TensorData& input,
                             const Nv12Layout& layout,
                             std::vector<uint8_t>* output) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  ValidateInput(input, layout, output);
  output->assign(Nv12BufferSize(layout), 0);
  const int input_height = static_cast<int>(input.shape.dim(2));
  const int input_width = static_cast<int>(input.shape.dim(3));
  const std::size_t plane = static_cast<std::size_t>(input_height) * input_width;
  const auto* source = reinterpret_cast<const uint16_t*>(input.bytes.data());

  for (int y = 0; y < layout.height; ++y) {
    int x = 0;
    for (; x + 8 <= layout.width; x += 8) {
      const std::size_t input_offset = static_cast<std::size_t>(y) * input_width + x;
      const std::size_t output_offset = static_cast<std::size_t>(y) * layout.width_stride + x;
      vst1_u8(output->data() + output_offset, NormalizedHalf8ToBytes(source + input_offset));
    }
    for (; x < layout.width; ++x) {
      const std::size_t input_offset = static_cast<std::size_t>(y) * input_width + x;
      const std::size_t output_offset = static_cast<std::size_t>(y) * layout.width_stride + x;
      (*output)[output_offset] = NormalizedToByte(codec::HalfBitsToFloat(source[input_offset]));
    }
  }

  const std::size_t uv_base =
      static_cast<std::size_t>(layout.width_stride) * layout.height_stride;
  for (int y = 0; y < layout.height; y += 2) {
    int x = 0;
    for (; x + 8 <= layout.width; x += 8) {
      const std::size_t row0 = static_cast<std::size_t>(y) * input_width + x;
      const std::size_t row1 = static_cast<std::size_t>(y + 1) * input_width + x;
      uint16_t u[4];
      uint16_t v[4];
      vst1_u16(u, NormalizedFloat4ToU16(
                        AverageHalf2x2(source + plane + row0, source + plane + row1)));
      vst1_u16(v, NormalizedFloat4ToU16(
                        AverageHalf2x2(source + 2 * plane + row0, source + 2 * plane + row1)));
      const std::size_t uv_offset =
          uv_base + static_cast<std::size_t>(y / 2) * layout.width_stride + x;
      for (int pair = 0; pair < 4; ++pair) {
        (*output)[uv_offset + 2 * pair] = static_cast<uint8_t>(u[pair]);
        (*output)[uv_offset + 2 * pair + 1] = static_cast<uint8_t>(v[pair]);
      }
    }
    for (; x < layout.width; x += 2) {
      float u_sum = 0.0F;
      float v_sum = 0.0F;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const std::size_t offset = static_cast<std::size_t>(y + dy) * input_width + x + dx;
          u_sum += codec::HalfBitsToFloat(source[plane + offset]);
          v_sum += codec::HalfBitsToFloat(source[2 * plane + offset]);
        }
      }
      const std::size_t uv_offset =
          uv_base + static_cast<std::size_t>(y / 2) * layout.width_stride + x;
      (*output)[uv_offset] = NormalizedToByte(u_sum * 0.25F);
      (*output)[uv_offset + 1] = NormalizedToByte(v_sum * 0.25F);
    }
  }
#else
  ConvertFp16Yuv444ToNv12Scalar(input, layout, output);
#endif
}

}  // namespace mlvc::io
