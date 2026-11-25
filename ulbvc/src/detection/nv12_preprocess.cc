#include "ulbvc/detection/nv12_preprocess.h"

#include <algorithm>
#include <opencv2/core.hpp>
#include <stdexcept>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include "ulbvc/codec/tensor_data.h"

namespace ulbvc::detection {
namespace {
constexpr int kY = 1220542, kUB = 2116026, kUG = -409993, kVG = -852492, kVR = 1673527;
uint16_t Normalize(int value) {
  const int pixel = std::clamp((value + (1 << 19)) >> 20, 0, 255);
  return codec::FloatToHalfBits(static_cast<float>(pixel) / 255.0f);
}
#ifdef __aarch64__
__attribute__((always_inline)) inline void StoreNormalized(int32x4_t first, int32x4_t second,
                                                           uint16_t* output) {
  const int32x4_t rounding = vdupq_n_s32(1 << 19);
  first = vshrq_n_s32(vaddq_s32(first, rounding), 20);
  second = vshrq_n_s32(vaddq_s32(second, rounding), 20);
  const uint8x8_t rgb = vqmovn_u16(vcombine_u16(vqmovun_s32(first), vqmovun_s32(second)));
  const uint16x8_t wide = vmovl_u8(rgb);
  const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
  const float32x4_t low = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(wide))), scale);
  const float32x4_t high = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(wide))), scale);
  vst1q_u16(output, vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(low), vcvt_f16_f32(high))));
}
#endif
}  // namespace

void PackNv12ToRgbFp16(const uint8_t* y, std::size_t y_stride, const uint8_t* uv,
                       std::size_t uv_stride, int width, int height, int input_width,
                       int input_height, int left, int top, uint16_t* output) {
  if (!y || !uv || !output || width <= 0 || height <= 0 || width % 2 || height % 2 ||
      y_stride < static_cast<std::size_t>(width) || uv_stride < static_cast<std::size_t>(width) ||
      left < 0 || top < 0 || input_width < width || input_height < height ||
      left > input_width - width || top > input_height - height)
    throw std::invalid_argument("invalid detection NV12/letterbox layout");
  const std::size_t plane = static_cast<std::size_t>(input_width) * input_height;
  const auto rows = [&](const cv::Range& range) {
    for (int row = range.start; row < range.end; ++row) {
      const auto* luma = y + row * y_stride;
      const auto* chroma = uv + (row / 2) * uv_stride;
      const std::size_t offset = static_cast<std::size_t>(row + top) * input_width + left;
      int x = 0;
#ifdef __aarch64__
      for (; x + 16 <= width; x += 16) {
        const uint8x8x2_t pairs = vld2_u8(chroma + x);
        const uint8x8x2_t u = vzip_u8(pairs.val[0], pairs.val[0]);
        const uint8x8x2_t v = vzip_u8(pairs.val[1], pairs.val[1]);
        for (int half = 0; half < 2; ++half) {
          const int16x8_t yy =
              vmaxq_s16(vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(luma + x + half * 8))),
                                  vdupq_n_s16(16)),
                        vdupq_n_s16(0));
          const int16x8_t uu =
              vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u.val[half])), vdupq_n_s16(128));
          const int16x8_t vv =
              vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v.val[half])), vdupq_n_s16(128));
          const int32x4_t yl = vmulq_n_s32(vmovl_s16(vget_low_s16(yy)), kY);
          const int32x4_t yh = vmulq_n_s32(vmovl_s16(vget_high_s16(yy)), kY);
          const int32x4_t ul = vmovl_s16(vget_low_s16(uu)), uh = vmovl_s16(vget_high_s16(uu));
          const int32x4_t vl = vmovl_s16(vget_low_s16(vv)), vh = vmovl_s16(vget_high_s16(vv));
          auto* destination = output + offset + x + half * 8;
          StoreNormalized(vmlaq_n_s32(yl, vl, kVR), vmlaq_n_s32(yh, vh, kVR), destination);
          StoreNormalized(vmlaq_n_s32(vmlaq_n_s32(yl, ul, kUG), vl, kVG),
                          vmlaq_n_s32(vmlaq_n_s32(yh, uh, kUG), vh, kVG), destination + plane);
          StoreNormalized(vmlaq_n_s32(yl, ul, kUB), vmlaq_n_s32(yh, uh, kUB),
                          destination + 2 * plane);
        }
      }
#endif
      for (; x < width; ++x) {
        const int yy = std::max(0, static_cast<int>(luma[x]) - 16) * kY;
        const int u = static_cast<int>(chroma[(x / 2) * 2]) - 128;
        const int v = static_cast<int>(chroma[(x / 2) * 2 + 1]) - 128;
        output[offset + x] = Normalize(yy + kVR * v);
        output[plane + offset + x] = Normalize(yy + kUG * u + kVG * v);
        output[2 * plane + offset + x] = Normalize(yy + kUB * u);
      }
    }
  };
  cv::parallel_for_(cv::Range(0, height), rows);
}
}  // namespace ulbvc::detection
