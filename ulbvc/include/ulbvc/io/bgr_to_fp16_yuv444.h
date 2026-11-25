#ifndef ULBVC_IO_BGR_TO_FP16_YUV444_H_
#define ULBVC_IO_BGR_TO_FP16_YUV444_H_
#include <cstddef>
#include <cstdint>

namespace ulbvc::io {
void ConvertBgrToFp16Yuv444(const uint8_t* bgr, std::size_t stride, int width, int height,
                            int padded_width, int padded_height, uint16_t* output);
}
#endif
