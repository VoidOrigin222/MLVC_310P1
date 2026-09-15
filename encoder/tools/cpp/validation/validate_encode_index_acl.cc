#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "mlvc/core/buffer.h"
#include "mlvc/core/status.h"
#include "mlvc/entropy/entropy_codec.h"
#include "mlvc/runtime/acl_runtime.h"
#include "mlvc/runtime/encode_index_acl.h"

namespace {

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " [--device 0] [--count 4096] [--force-zero-thres 0.6]"
               " [--symbol-dtype int8|fp16|both]\n";
}

uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000U;
  const uint32_t float_exponent = (bits >> 23) & 0xffU;
  uint32_t mantissa = bits & 0x7fffffU;
  if (float_exponent == 0xffU) {
    return static_cast<uint16_t>(sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
  }

  int32_t exponent = static_cast<int32_t>(float_exponent) - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000U;
    const int shift = 14 - exponent;
    uint32_t half_mantissa = mantissa >> shift;
    const uint32_t remainder = mantissa & ((uint32_t{1} << shift) - 1U);
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0U)) {
      ++half_mantissa;
    }
    return static_cast<uint16_t>(sign | half_mantissa);
  }

  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }

  uint32_t half_mantissa = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0U)) {
    ++half_mantissa;
    if (half_mantissa == 0x400U) {
      half_mantissa = 0;
      ++exponent;
      if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
      }
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | half_mantissa);
}

float HalfBitsToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
  uint32_t exponent = (value >> 10) & 0x1fU;
  uint32_t mantissa = value & 0x03ffU;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffU;
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float output = 0.0f;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

uint8_t ScaleIndex(float scale) {
  const float log_scale_min = std::log(mlvc::kScaleMin);
  const float log_scale_max = std::log(mlvc::kScaleMax);
  const float log_scale_step = (log_scale_max - log_scale_min) / (mlvc::kScaleLevel - 1);
  const float clipped = std::min(std::max(scale, mlvc::kScaleMin), mlvc::kScaleMax);
  return static_cast<uint8_t>((std::log(clipped) - log_scale_min) / log_scale_step);
}

int32_t SymbolFromFp16(uint16_t bits) {
  const float value = HalfBitsToFloat(bits);
  return value >= 0.0f ? static_cast<int32_t>(value + 0.5f) : static_cast<int32_t>(value - 0.5f);
}

void CheckAclStatus(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw mlvc::Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

std::vector<uint16_t> MakeScaleBits(std::size_t count) {
  const float pattern[] = {0.05f, 0.11f, 0.16f, 0.24f, 0.43f, 0.60f,
                           0.95f, 1.30f, 3.75f, 16.0f, 18.0f};
  std::vector<uint16_t> scales(count);
  for (std::size_t i = 0; i < count; ++i) {
    scales[i] = FloatToHalfBits(pattern[i % (sizeof(pattern) / sizeof(pattern[0]))]);
  }
  return scales;
}

std::vector<int8_t> MakeInt8Symbols(std::size_t count) {
  std::vector<int8_t> symbols(count);
  for (std::size_t i = 0; i < count; ++i) {
    symbols[i] = static_cast<int8_t>(static_cast<int>((i * 7 + 3) % 17) - 8);
  }
  return symbols;
}

std::vector<uint16_t> MakeFp16Symbols(std::size_t count) {
  std::vector<uint16_t> symbols(count);
  for (std::size_t i = 0; i < count; ++i) {
    const int value = static_cast<int>((i * 7 + 3) % 17) - 8;
    symbols[i] = FloatToHalfBits(static_cast<float>(value));
  }
  return symbols;
}

template <typename SymbolReader>
void MakeReference(std::size_t count, const std::vector<uint16_t>& scales, float force_zero_thres,
                   SymbolReader symbol_reader, std::vector<int16_t>* combined,
                   std::vector<uint8_t>* keep_mask) {
  combined->resize(count);
  keep_mask->resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    const float scale = HalfBitsToFloat(scales[i]);
    const float clipped = std::min(std::max(scale, mlvc::kScaleMin), mlvc::kScaleMax);
    const bool keep = force_zero_thres < 0.0f || clipped > force_zero_thres;
    const int32_t symbol = symbol_reader(i);
    const int32_t index = static_cast<int32_t>(ScaleIndex(scale));
    (*combined)[i] = static_cast<int16_t>((symbol << 8) + index);
    (*keep_mask)[i] = keep ? 1U : 0U;
  }
}

void RunCase(const std::string& symbol_dtype, std::size_t count, float force_zero_thres,
             void* stream) {
  const std::vector<int64_t> shape = {1, 1, 1, static_cast<int64_t>(count)};
  const std::vector<uint16_t> scales = MakeScaleBits(count);
  std::vector<int16_t> expected_combined;
  std::vector<uint8_t> expected_keep;

  std::vector<uint8_t> host_symbols;
  mlvc::DataType dtype = mlvc::DataType::kInt8;
  if (symbol_dtype == "int8") {
    const std::vector<int8_t> symbols = MakeInt8Symbols(count);
    host_symbols.resize(symbols.size() * sizeof(int8_t));
    std::memcpy(host_symbols.data(), symbols.data(), host_symbols.size());
    MakeReference(
        count, scales, force_zero_thres,
        [&](std::size_t i) { return static_cast<int32_t>(symbols[i]); }, &expected_combined,
        &expected_keep);
  } else if (symbol_dtype == "fp16") {
    const std::vector<uint16_t> symbols = MakeFp16Symbols(count);
    host_symbols.resize(symbols.size() * sizeof(uint16_t));
    std::memcpy(host_symbols.data(), symbols.data(), host_symbols.size());
    dtype = mlvc::DataType::kFloat16;
    MakeReference(
        count, scales, force_zero_thres, [&](std::size_t i) { return SymbolFromFp16(symbols[i]); },
        &expected_combined, &expected_keep);
  } else {
    throw mlvc::Error("unsupported symbol dtype: " + symbol_dtype);
  }

  mlvc::AclBuffer device_symbols(host_symbols.size());
  mlvc::AclBuffer device_scales(scales.size() * sizeof(uint16_t));
  mlvc::AclBuffer device_combined(count * sizeof(int16_t));
  mlvc::AclBuffer device_keep(count * sizeof(uint8_t));
  CheckAclStatus(aclrtMemcpy(device_symbols.data(), device_symbols.bytes(), host_symbols.data(),
                             host_symbols.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy symbols H2D");
  CheckAclStatus(aclrtMemcpy(device_scales.data(), device_scales.bytes(), scales.data(),
                             scales.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy scales H2D");

  const mlvc::EncodeIndexAclResult result = mlvc::BuildIndexEncodeAcl(
      device_symbols.data(), dtype, device_scales.data(), mlvc::DataType::kFloat16, shape.data(),
      shape.size(), count, force_zero_thres, static_cast<int16_t*>(device_combined.data()),
      static_cast<uint8_t*>(device_keep.data()), stream);

  std::vector<int16_t> actual_combined(count);
  std::vector<uint8_t> actual_keep(count);
  CheckAclStatus(
      aclrtMemcpy(actual_combined.data(), actual_combined.size() * sizeof(int16_t),
                  device_combined.data(), device_combined.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
      "aclrtMemcpy combined D2H");
  CheckAclStatus(aclrtMemcpy(actual_keep.data(), actual_keep.size(), device_keep.data(),
                             device_keep.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy keep D2H");

  std::size_t combined_mismatches = 0;
  std::size_t keep_mismatches = 0;
  std::size_t first_mismatch = count;
  for (std::size_t i = 0; i < count; ++i) {
    if (actual_combined[i] != expected_combined[i]) {
      ++combined_mismatches;
      first_mismatch = std::min(first_mismatch, i);
    }
    if (actual_keep[i] != expected_keep[i]) {
      ++keep_mismatches;
      first_mismatch = std::min(first_mismatch, i);
    }
  }
  if (combined_mismatches != 0 || keep_mismatches != 0) {
    std::cerr << "first_mismatch=" << first_mismatch
              << " expected_combined=" << expected_combined[first_mismatch]
              << " actual_combined=" << actual_combined[first_mismatch]
              << " expected_keep=" << static_cast<int>(expected_keep[first_mismatch])
              << " actual_keep=" << static_cast<int>(actual_keep[first_mismatch]) << "\n";
    throw mlvc::Error("ACL encode-index mismatch for " + symbol_dtype);
  }

  std::cout << "case=" << symbol_dtype << " elements=" << result.elements
            << " force_zero_thres=" << force_zero_thres << " status=ok\n";
}

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  std::size_t count = 4096;
  float force_zero_thres = 0.6f;
  std::string symbol_dtype = "both";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) {
      device = std::stoi(argv[++i]);
    } else if (arg == "--count" && i + 1 < argc) {
      count = static_cast<std::size_t>(std::stoull(argv[++i]));
    } else if (arg == "--force-zero-thres" && i + 1 < argc) {
      force_zero_thres = std::stof(argv[++i]);
    } else if (arg == "--symbol-dtype" && i + 1 < argc) {
      symbol_dtype = argv[++i];
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }

  try {
    mlvc::Check(count > 0, "count must be positive");
    mlvc::AclRuntime runtime(device);
    mlvc::Check(mlvc::EncodeIndexAclAvailable(),
                "MlvcBuildIndexEncode ACLNN custom op is not available");
    if (symbol_dtype == "both") {
      RunCase("int8", count, force_zero_thres, runtime.stream());
      RunCase("fp16", count, force_zero_thres, runtime.stream());
    } else {
      RunCase(symbol_dtype, count, force_zero_thres, runtime.stream());
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "validate_encode_index_acl failed: " << error.what() << "\n";
    return 1;
  }
}
