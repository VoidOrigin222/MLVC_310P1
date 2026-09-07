#ifndef MLVC_IO_FP16_YUV444_TO_NV12_H_
#define MLVC_IO_FP16_YUV444_TO_NV12_H_

#include <mlvc/codec/tensor_data.h>

#include <cstddef>
#include <vector>

namespace mlvc::io {

struct Nv12Layout {
  int width = 0;
  int height = 0;
  int width_stride = 0;
  int height_stride = 0;
};

std::size_t Nv12BufferSize(const Nv12Layout& layout);
bool Fp16Yuv444ToNv12UsesNeon();
void ConvertFp16Yuv444ToNv12Scalar(const codec::TensorData& input,
                                   const Nv12Layout& layout,
                                   std::vector<uint8_t>* output);
void ConvertFp16Yuv444ToNv12(const codec::TensorData& input,
                             const Nv12Layout& layout,
                             std::vector<uint8_t>* output);

}  // namespace mlvc::io

#endif  // MLVC_IO_FP16_YUV444_TO_NV12_H_
