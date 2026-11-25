#ifndef ULBVC_CODEC_DETAIL_TENSOR_UTILS_H_
#define ULBVC_CODEC_DETAIL_TENSOR_UTILS_H_

#include <ulbvc/codec/detail/stage/stage_types.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/tensor.h>

#include <cstdint>
#include <vector>

namespace ulbvc::codec {

void Fp16ToFloat(const TensorData& tensor, std::vector<float>* values);
void Fp16ToFloat(const TensorBytesView& tensor, std::vector<float>* values);
void TensorToInt8Symbols(const TensorData& tensor, std::vector<int8_t>* symbols);
void TensorToInt8Symbols(const TensorBytesView& tensor, std::vector<int8_t>* symbols);
void FloatToFp16TensorFromInt8(const std::vector<int8_t>& symbols, const ulbvc::TensorShape& shape,
                               TensorData* tensor);
void FloatToFp16Tensor(const std::vector<float>& values, const std::vector<int64_t>& shape,
                       TensorData* tensor);
void FloatToFp16Tensor(const std::vector<float>& values, const ulbvc::TensorShape& shape,
                       TensorData* tensor);
void ApplyChannelQuantStep(const std::vector<float>& quant_step, int channels, int height,
                           int width, std::vector<float>* values);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_TENSOR_UTILS_H_
