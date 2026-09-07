#include <mlvc/codec/detail/tensor/tensor_utils.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <mlvc/codec/detail/profile/codec_profile.h>
#include <mlvc/codec/detail/quantization/qscale_cache.h>
#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc::codec {

uint32_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  const uint32_t float_exponent = (bits >> 23) & 0xff;
  uint32_t mantissa = bits & 0x7fffff;

  if (float_exponent == 0xff) {
    if (mantissa == 0) {
      return sign | 0x7c00;
    }
    return sign | 0x7e00;
  }

  int32_t exponent = static_cast<int32_t>(float_exponent) - 127 + 15;

  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000;
    const int shift = 14 - exponent;
    uint32_t half_mantissa = mantissa >> shift;
    const uint32_t remainder = mantissa & ((uint32_t{1} << shift) - 1);
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0)) {
      ++half_mantissa;
    }
    return sign | half_mantissa;
  }

  if (exponent >= 31) {
    return sign | 0x7c00U;
  }

  uint32_t half_mantissa = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0)) {
    ++half_mantissa;
    if (half_mantissa == 0x400U) {
      half_mantissa = 0;
      ++exponent;
      if (exponent >= 31) {
        return sign | 0x7c00U;
      }
    }
  }
  return sign | (static_cast<uint32_t>(exponent) << 10) | half_mantissa;
}

float HalfBitsToFloat(uint16_t value) {
  const uint32_t sign = (value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 0x1f;
  uint32_t mantissa = value & 0x03ff;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ff;
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }

  float output = 0.0f;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

float RoundToFp16(float value) {
  return HalfBitsToFloat(static_cast<uint16_t>(FloatToHalfBits(value)));
}

TensorData MakeTensor(const std::vector<int64_t>& shape, mlvc::DataType dtype) {
  TensorData tensor;
  tensor.shape = mlvc::TensorShape(shape);
  tensor.dtype = dtype;
  tensor.bytes.resize(tensor.shape.NumElements() * mlvc::ElementSize(dtype));
  return tensor;
}

TensorData MakeTensorLike(const mlvc::TensorSpec& spec) {
  return MakeTensor(spec.shape, spec.dtype);
}

TensorData CloneTensor(const TensorData& tensor) {
  TensorData clone;
  clone.shape = tensor.shape;
  clone.dtype = tensor.dtype;
  clone.bytes = tensor.bytes;
  return clone;
}

void CloneTensorInto(const TensorData& tensor, TensorData* output) {
  mlvc::Check(output != nullptr, "clone tensor output is required");
  if (output->dtype != tensor.dtype || output->shape.dims() != tensor.shape.dims()) {
    output->shape = tensor.shape;
    output->dtype = tensor.dtype;
  }
  output->bytes.resize(tensor.bytes.size());
  std::memcpy(output->bytes.data(), tensor.bytes.data(), tensor.bytes.size());
}

const mlvc::TensorSpec& OutputSpec(const mlvc::ModelRecord& record, std::string_view name) {
  for (const mlvc::TensorSpec& output : record.outputs) {
    if (output.name == name) {
      return output;
    }
  }
  throw mlvc::Error("missing output spec: " + record.name + "." + std::string(name));
}

TensorData MakeFp16Tensor(const std::vector<int64_t>& shape, float value) {
  TensorData tensor = MakeTensor(shape, mlvc::DataType::kFloat16);
  FillFp16Tensor(value, &tensor);
  return tensor;
}

void FillFp16Tensor(float value, TensorData* tensor) {
  mlvc::Check(tensor != nullptr, "fill tensor output is required");
  mlvc::Check(tensor->dtype == mlvc::DataType::kFloat16, "fill tensor expects fp16");
  const uint16_t half = static_cast<uint16_t>(FloatToHalfBits(value));
  auto* data = reinterpret_cast<uint16_t*>(tensor->bytes.data());
  std::fill(data, data + tensor->Elements(), half);
}

TensorData ReadTensorFile(const std::filesystem::path& path, const std::vector<int64_t>& shape,
                          mlvc::DataType dtype) {
  TensorData tensor = MakeTensor(shape, dtype);
  ReadTensorFileInto(path, &tensor);
  return tensor;
}

void ReadTensorFileInto(const std::filesystem::path& path, TensorData* tensor) {
  mlvc::Check(tensor != nullptr, "tensor input output is required");
  std::ifstream input(path, std::ios::binary);
  mlvc::Check(input.good(), "failed to open tensor input: " + path.string());
  input.read(reinterpret_cast<char*>(tensor->bytes.data()),
             static_cast<std::streamsize>(tensor->bytes.size()));
  mlvc::Check(input.good(), "failed to read tensor input: " + path.string());
  input.peek();
  mlvc::Check(input.eof(), "tensor input has trailing bytes: " + path.string());
}

void WriteTensorFile(const std::filesystem::path& path, const TensorData& tensor) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary);
  mlvc::Check(output.good(), "failed to open tensor output: " + path.string());
  output.write(reinterpret_cast<const char*>(tensor.bytes.data()),
               static_cast<std::streamsize>(tensor.bytes.size()));
  mlvc::Check(output.good(), "failed to write tensor output: " + path.string());
}

TensorData MakeQScaleTensor(const mlvc::RuntimeSidecar& sidecar, const std::string& name, int qp,
                            const std::vector<int64_t>& shape) {
  TensorData tensor = MakeTensor(shape, mlvc::DataType::kFloat16);
  const uint16_t* row = sidecar.QScaleData(name, qp);
  std::memcpy(tensor.bytes.data(), row, tensor.bytes.size());
  return tensor;
}

void Fp16ToFloat(const TensorData& tensor, std::vector<float>* values) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(tensor.dtype == mlvc::DataType::kFloat16, "tensor is not fp16");
  values->resize(tensor.Elements());
  const auto* input = reinterpret_cast<const uint16_t*>(tensor.bytes.data());
  for (std::size_t i = 0; i < values->size(); ++i) {
    (*values)[i] = HalfBitsToFloat(input[i]);
  }
}

void TensorToInt8Symbols(const TensorData& tensor, std::vector<int8_t>* symbols) {
  MLVC_PROFILE_RANGE_FUNCTION();
  symbols->resize(tensor.Elements());
  if (tensor.dtype == mlvc::DataType::kInt8) {
    const auto* input = reinterpret_cast<const int8_t*>(tensor.bytes.data());
    std::copy(input, input + tensor.Elements(), symbols->begin());
    return;
  }
  mlvc::Check(tensor.dtype == mlvc::DataType::kFloat16, "tensor is not int8 or fp16");
  const auto* input = reinterpret_cast<const uint16_t*>(tensor.bytes.data());
  for (std::size_t i = 0; i < symbols->size(); ++i) {
    (*symbols)[i] = static_cast<int8_t>(HalfBitsToFloat(input[i]));
  }
}

void TensorToInt8Symbols(const TensorBytesView& tensor, std::vector<int8_t>* symbols) {
  MLVC_PROFILE_RANGE_FUNCTION();
  symbols->resize(tensor.Elements());
  if (tensor.dtype == mlvc::DataType::kInt8) {
    const auto* input = reinterpret_cast<const int8_t*>(tensor.bytes);
    std::copy(input, input + tensor.Elements(), symbols->begin());
    return;
  }
  mlvc::Check(tensor.dtype == mlvc::DataType::kFloat16, "tensor is not int8 or fp16");
  const auto* input = reinterpret_cast<const uint16_t*>(tensor.bytes);
  for (std::size_t i = 0; i < symbols->size(); ++i) {
    (*symbols)[i] = static_cast<int8_t>(HalfBitsToFloat(input[i]));
  }
}

void FloatToFp16TensorFromInt8(const std::vector<int8_t>& symbols, const mlvc::TensorShape& shape,
                               TensorData* tensor) {
  MLVC_PROFILE_RANGE_FUNCTION();
  tensor->shape = shape;
  tensor->dtype = mlvc::DataType::kFloat16;
  tensor->bytes.resize(tensor->shape.NumElements() * mlvc::ElementSize(tensor->dtype));
  auto* output = reinterpret_cast<uint16_t*>(tensor->bytes.data());
  for (std::size_t i = 0; i < symbols.size(); ++i) {
    output[i] = static_cast<uint16_t>(FloatToHalfBits(static_cast<float>(symbols[i])));
  }
}

void FloatToFp16Tensor(const std::vector<float>& values, const std::vector<int64_t>& shape,
                       TensorData* tensor) {
  MLVC_PROFILE_RANGE_FUNCTION();
  tensor->shape = mlvc::TensorShape(shape);
  tensor->dtype = mlvc::DataType::kFloat16;
  tensor->bytes.resize(tensor->shape.NumElements() * mlvc::ElementSize(tensor->dtype));
  auto* output = reinterpret_cast<uint16_t*>(tensor->bytes.data());
  for (std::size_t i = 0; i < values.size(); ++i) {
    output[i] = static_cast<uint16_t>(FloatToHalfBits(values[i]));
  }
}

void FloatToFp16Tensor(const std::vector<float>& values, const mlvc::TensorShape& shape,
                       TensorData* tensor) {
  MLVC_PROFILE_RANGE_FUNCTION();
  tensor->shape = shape;
  tensor->dtype = mlvc::DataType::kFloat16;
  tensor->bytes.resize(tensor->shape.NumElements() * mlvc::ElementSize(tensor->dtype));
  auto* output = reinterpret_cast<uint16_t*>(tensor->bytes.data());
  for (std::size_t i = 0; i < values.size(); ++i) {
    output[i] = static_cast<uint16_t>(FloatToHalfBits(values[i]));
  }
}

void Fp16ToFloat(const TensorBytesView& tensor, std::vector<float>* values) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(tensor.dtype == mlvc::DataType::kFloat16, "tensor is not fp16");
  values->resize(tensor.Elements());
  const auto* input = reinterpret_cast<const uint16_t*>(tensor.bytes);
  for (std::size_t i = 0; i < values->size(); ++i) {
    (*values)[i] = HalfBitsToFloat(input[i]);
  }
}

void ApplyChannelQuantStep(const std::vector<float>& quant_step, int channels, int height,
                           int width, std::vector<float>* values) {
  MLVC_PROFILE_RANGE_FUNCTION();
  const int quant_channels = static_cast<int>(quant_step.size()) / (height * width);
  mlvc::Check(static_cast<int>(quant_step.size()) == quant_channels * height * width,
              "decoder quant step shape is not contiguous NCHW");
  mlvc::Check(quant_channels == 1 || quant_channels == channels,
              "decoder quant step channel count mismatch");
  for (int c = 0; c < channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int quant_channel = quant_channels == 1 ? 0 : c;
        const float q = quant_step[((quant_channel * height) + y) * width + x];
        const int index = ((c * height) + y) * width + x;
        (*values)[index] = RoundToFp16((*values)[index] * q);
      }
    }
  }
}

}  // namespace mlvc::codec
