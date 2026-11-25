#include "ulbvc/io/bgr_to_fp16_yuv444.h"

#include <algorithm>
#include <opencv2/core.hpp>
#include <stdexcept>

#include "ulbvc/codec/tensor_data.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace ulbvc::io {
namespace {
constexpr float kKr = 0.2126f, kKg = 0.7152f, kKb = 0.0722f;
}

void ConvertBgrToFp16Yuv444(const uint8_t* bgr, std::size_t stride, int width, int height,
                            int padded_width, int padded_height, uint16_t* output) {
  if (!bgr || !output || width <= 0 || height <= 0 || padded_width < width ||
      padded_height < height || stride < static_cast<std::size_t>(width) * 3)
    throw std::invalid_argument("invalid BGR/FP16 source layout");
  const std::size_t plane = static_cast<std::size_t>(padded_width) * padded_height;
  const auto rows = [&](int first, int last) {
    for (int y = first; y < last; ++y) {
      const auto* row = bgr + std::min(y, height - 1) * stride;
      const auto offset = static_cast<std::size_t>(y) * padded_width;
      int x = 0;
#ifdef __aarch64__
      const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
      const float32x4_t half = vdupq_n_f32(0.5f);
      for (; x + 16 <= width; x += 16) {
        const uint8x16x3_t pixels = vld3q_u8(row + x * 3);
        for (int part = 0; part < 4; ++part) {
          float32x4_t channels[3];
          for (int c = 0; c < 3; ++c) {
            const uint16x8_t values =
                vmovl_u8(part < 2 ? vget_low_u8(pixels.val[c]) : vget_high_u8(pixels.val[c]));
            const uint32x4_t wide =
                vmovl_u16(part % 2 == 0 ? vget_low_u16(values) : vget_high_u16(values));
            channels[c] = vmulq_f32(vcvtq_f32_u32(wide), scale);
          }
          const auto blue = channels[0], green = channels[1], red = channels[2];
          const float32x4_t yy = vaddq_f32(
              vaddq_f32(vmulq_n_f32(red, kKr), vmulq_n_f32(green, kKg)), vmulq_n_f32(blue, kKb));
          const float32x4_t cb =
              vaddq_f32(vmulq_n_f32(vsubq_f32(blue, yy), 0.5f / (1.0f - kKb)), half);
          const float32x4_t cr =
              vaddq_f32(vmulq_n_f32(vsubq_f32(red, yy), 0.5f / (1.0f - kKr)), half);
          auto* destination = output + offset + x + part * 4;
          vst1_u16(destination, vreinterpret_u16_f16(vcvt_f16_f32(yy)));
          vst1_u16(destination + plane, vreinterpret_u16_f16(vcvt_f16_f32(cb)));
          vst1_u16(destination + 2 * plane, vreinterpret_u16_f16(vcvt_f16_f32(cr)));
        }
      }
#endif
      for (; x < padded_width; ++x) {
        const auto* pixel = row + std::min(x, width - 1) * 3;
        const float b = pixel[0] / 255.0f, g = pixel[1] / 255.0f, r = pixel[2] / 255.0f;
        const float yy = kKr * r + kKg * g + kKb * b;
        const float cb = 0.5f * (b - yy) / (1 - kKb) + 0.5f;
        const float cr = 0.5f * (r - yy) / (1 - kKr) + 0.5f;
        output[offset + x] = codec::FloatToHalfBits(yy);
        output[plane + offset + x] = codec::FloatToHalfBits(cb);
        output[2 * plane + offset + x] = codec::FloatToHalfBits(cr);
      }
    }
  };
  cv::parallel_for_(cv::Range(0, padded_height),
                    [&](const cv::Range& range) { rows(range.start, range.end); });
}
}  // namespace ulbvc::io
