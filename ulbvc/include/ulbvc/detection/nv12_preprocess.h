#ifndef ULBVC_DETECTION_NV12_PREPROCESS_H_
#define ULBVC_DETECTION_NV12_PREPROCESS_H_

#include <cstddef>
#include <cstdint>

namespace ulbvc::detection {

// BT.601 limited-range NV12 -> RGB uint8 rounding -> normalized FP16 NCHW.
// Writes only the visible rectangle, preserving the caller's letterbox padding.
void PackNv12ToRgbFp16(const uint8_t* y, std::size_t y_stride, const uint8_t* uv,
                       std::size_t uv_stride, int width, int height, int input_width,
                       int input_height, int left, int top, uint16_t* output);

}  // namespace ulbvc::detection
#endif
