#ifndef ULBVC_CODEC_TENSOR_UTILS_H_
#define ULBVC_CODEC_TENSOR_UTILS_H_

#include <ulbvc/codec/tensor_data.h>

#include <cstdint>
#include <vector>

namespace ulbvc::codec {

void Fp16ToFloat(const TensorData& tensor, std::vector<float>* values);
void TensorToInt8Symbols(const TensorData& tensor, std::vector<int8_t>* symbols);
void FloatToFp16TensorFromInt8(const std::vector<int8_t>& symbols, const ulbvc::TensorShape& shape,
                               TensorData* tensor);
void FloatToFp16Tensor(const std::vector<float>& values, const std::vector<int64_t>& shape,
                       TensorData* tensor);
void FloatToFp16Tensor(const std::vector<float>& values, const ulbvc::TensorShape& shape,
                       TensorData* tensor);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_TENSOR_UTILS_H_
