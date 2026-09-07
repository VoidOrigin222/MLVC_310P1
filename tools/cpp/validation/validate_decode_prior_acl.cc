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
#include "mlvc/runtime/acl_runtime.h"
#include "mlvc/runtime/decode_prior_acl.h"

namespace {

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " [--device 0] [--channels 16] [--height 7] [--width 9]\n";
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

float RoundToFp16(float value) { return HalfBitsToFloat(FloatToHalfBits(value)); }

int Offset(int c, int y, int x, int height, int width) { return ((c * height) + y) * width + x; }

int MicroMask4(int mask_index, int chunk, int y, int x) {
  static constexpr int patterns[4][4][2][2] = {
      {{{1, 0}, {0, 0}}, {{0, 1}, {0, 0}}, {{0, 0}, {1, 0}}, {{0, 0}, {0, 1}}},
      {{{0, 0}, {0, 1}}, {{0, 0}, {1, 0}}, {{0, 1}, {0, 0}}, {{1, 0}, {0, 0}}},
      {{{0, 0}, {1, 0}}, {{0, 0}, {0, 1}}, {{1, 0}, {0, 0}}, {{0, 1}, {0, 0}}},
      {{{0, 1}, {0, 0}}, {{1, 0}, {0, 0}}, {{0, 0}, {0, 1}}, {{0, 0}, {1, 0}}},
  };
  return patterns[mask_index][chunk][y & 1][x & 1];
}

int MicroMask2(int mask_index, int chunk, int y, int x) {
  static constexpr int patterns[2][2][2][2] = {
      {{{1, 0}, {0, 1}}, {{0, 1}, {1, 0}}},
      {{{0, 1}, {1, 0}}, {{1, 0}, {0, 1}}},
  };
  return patterns[mask_index][chunk][y & 1][x & 1];
}

int SelectedChunk(int parts, int mask_index, int y, int x) {
  for (int chunk = 0; chunk < parts; ++chunk) {
    const int selected =
        parts == 4 ? MicroMask4(mask_index, chunk, y, x) : MicroMask2(mask_index, chunk, y, x);
    if (selected != 0) {
      return chunk;
    }
  }
  return 0;
}

void CheckAclStatus(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw mlvc::Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

std::vector<float> MakeFullFp32(int channels, int height, int width) {
  std::vector<float> values(static_cast<std::size_t>(channels) * height * width);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(static_cast<int>((i * 13 + 5) % 97) - 48) / 7.0f;
  }
  return values;
}

std::vector<uint16_t> Fp32ToFp16(const std::vector<float>& values) {
  std::vector<uint16_t> output(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    output[i] = FloatToHalfBits(values[i]);
  }
  return output;
}

std::vector<int8_t> MakeSymbols(std::size_t count) {
  std::vector<int8_t> output(count);
  for (std::size_t i = 0; i < output.size(); ++i) {
    output[i] = static_cast<int8_t>(static_cast<int>((i * 7 + 3) % 21) - 10);
  }
  return output;
}

std::vector<float> ReferenceSinglePartFp32(const std::vector<float>& full, int parts,
                                           int mask_index, int channels, int height, int width) {
  const int part_channels = channels / parts;
  std::vector<float> part(static_cast<std::size_t>(part_channels) * height * width);
  for (int c = 0; c < part_channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int chunk = SelectedChunk(parts, mask_index, y, x);
        part[Offset(c, y, x, height, width)] =
            full[Offset(chunk * part_channels + c, y, x, height, width)];
      }
    }
  }
  return part;
}

std::vector<uint16_t> ReferenceSinglePartFp16(const std::vector<uint16_t>& full, int parts,
                                              int mask_index, int channels, int height, int width) {
  const int part_channels = channels / parts;
  std::vector<uint16_t> part(static_cast<std::size_t>(part_channels) * height * width);
  for (int c = 0; c < part_channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int chunk = SelectedChunk(parts, mask_index, y, x);
        part[Offset(c, y, x, height, width)] =
            full[Offset(chunk * part_channels + c, y, x, height, width)];
      }
    }
  }
  return part;
}

void UpdateReferenceRestoreY(const std::vector<int8_t>& symbols, const std::vector<uint16_t>& means,
                             int parts, int mask_index, int channels, int height, int width,
                             std::vector<uint16_t>* output) {
  const int part_channels = channels / parts;
  for (int c = 0; c < part_channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int chunk = SelectedChunk(parts, mask_index, y, x);
        const int source = Offset(c, y, x, height, width);
        const int target = Offset(chunk * part_channels + c, y, x, height, width);
        (*output)[target] =
            FloatToHalfBits(static_cast<float>(symbols[source]) + HalfBitsToFloat(means[target]));
      }
    }
  }
}

std::vector<uint16_t> ReferenceRestoreY(const std::vector<int8_t>& symbols,
                                        const std::vector<uint16_t>& means, int parts,
                                        int mask_index, int channels, int height, int width) {
  std::vector<uint16_t> output(static_cast<std::size_t>(channels) * height * width, 0);
  UpdateReferenceRestoreY(symbols, means, parts, mask_index, channels, height, width, &output);
  return output;
}

std::vector<uint16_t> ReferenceApplyQuant(const std::vector<uint16_t>& quant, int quant_channels,
                                          const std::vector<uint16_t>& values, int channels,
                                          int height, int width) {
  std::vector<uint16_t> output(values.size());
  for (int c = 0; c < channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int quant_channel = quant_channels == 1 ? 0 : c;
        const int index = Offset(c, y, x, height, width);
        const int quant_index = Offset(quant_channel, y, x, height, width);
        output[index] =
            FloatToHalfBits(HalfBitsToFloat(values[index]) * HalfBitsToFloat(quant[quant_index]));
      }
    }
  }
  return output;
}

std::vector<uint16_t> ReferenceInt8ToFp16(const std::vector<int8_t>& symbols) {
  std::vector<uint16_t> output(symbols.size());
  for (std::size_t i = 0; i < symbols.size(); ++i) {
    output[i] = FloatToHalfBits(static_cast<float>(symbols[i]));
  }
  return output;
}

template <typename T>
void Upload(const std::vector<T>& host, mlvc::AclBuffer* device) {
  device->Allocate(host.size() * sizeof(T));
  CheckAclStatus(aclrtMemcpy(device->data(), device->bytes(), host.data(), device->bytes(),
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy H2D");
}

template <typename T>
std::vector<T> Download(const mlvc::AclBuffer& device, std::size_t count) {
  std::vector<T> host(count);
  CheckAclStatus(aclrtMemcpy(host.data(), host.size() * sizeof(T), device.data(), device.bytes(),
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy D2H");
  return host;
}

void CheckEqualBits(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& expected,
                    const std::string& name) {
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << name << " first_mismatch=" << i << " expected_bits=" << expected[i]
                << " actual_bits=" << actual[i] << " expected=" << HalfBitsToFloat(expected[i])
                << " actual=" << HalfBitsToFloat(actual[i]) << "\n";
      throw mlvc::Error(name + " mismatch");
    }
  }
}

void CheckEqualFp32(const std::vector<float>& actual, const std::vector<float>& expected,
                    const std::string& name) {
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << name << " first_mismatch=" << i << " expected=" << expected[i]
                << " actual=" << actual[i] << "\n";
      throw mlvc::Error(name + " mismatch");
    }
  }
}

void RunSinglePartCase(int parts, int mask_index, int channels, int height, int width,
                       void* stream) {
  const int part_channels = channels / parts;
  const std::size_t part_elements = static_cast<std::size_t>(part_channels) * height * width;
  const std::vector<float> full_fp32 = MakeFullFp32(channels, height, width);
  const std::vector<uint16_t> full_fp16 = Fp32ToFp16(full_fp32);

  mlvc::AclBuffer full_fp32_dev;
  mlvc::AclBuffer part_fp32_dev(part_elements * sizeof(float));
  Upload(full_fp32, &full_fp32_dev);
  mlvc::DecodePriorSinglePartAcl(full_fp32_dev.data(), mlvc::DataType::kFloat32, parts, mask_index,
                                 channels, height, width, part_fp32_dev.data(), stream);
  CheckEqualFp32(Download<float>(part_fp32_dev, part_elements),
                 ReferenceSinglePartFp32(full_fp32, parts, mask_index, channels, height, width),
                 "single_part_fp32");

  mlvc::AclBuffer full_fp16_dev;
  mlvc::AclBuffer part_fp16_dev(part_elements * sizeof(uint16_t));
  Upload(full_fp16, &full_fp16_dev);
  mlvc::DecodePriorSinglePartAcl(full_fp16_dev.data(), mlvc::DataType::kFloat16, parts, mask_index,
                                 channels, height, width, part_fp16_dev.data(), stream);
  CheckEqualBits(Download<uint16_t>(part_fp16_dev, part_elements),
                 ReferenceSinglePartFp16(full_fp16, parts, mask_index, channels, height, width),
                 "single_part_fp16");
}

void RunRestoreYCase(int parts, int mask_index, int channels, int height, int width, void* stream) {
  const int part_channels = channels / parts;
  const std::size_t part_elements = static_cast<std::size_t>(part_channels) * height * width;
  const std::size_t full_elements = static_cast<std::size_t>(channels) * height * width;
  const std::vector<int8_t> symbols = MakeSymbols(part_elements);
  const std::vector<uint16_t> means = Fp32ToFp16(MakeFullFp32(channels, height, width));

  mlvc::AclBuffer symbols_dev;
  mlvc::AclBuffer means_dev;
  mlvc::AclBuffer output_dev(full_elements * sizeof(uint16_t));
  Upload(symbols, &symbols_dev);
  Upload(means, &means_dev);
  std::vector<uint16_t> initial_output(full_elements, 0);
  CheckAclStatus(aclrtMemcpy(output_dev.data(), output_dev.bytes(), initial_output.data(),
                             output_dev.bytes(), ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy restore_y initial output H2D");
  mlvc::DecodePriorRestoreYAcl(static_cast<const int8_t*>(symbols_dev.data()), means_dev.data(),
                               parts, mask_index, channels, height, width, output_dev.data(),
                               stream);
  CheckEqualBits(Download<uint16_t>(output_dev, full_elements),
                 ReferenceRestoreY(symbols, means, parts, mask_index, channels, height, width),
                 "restore_y");
}

void RunRestoreYSequenceCase(int parts, int channels, int height, int width, void* stream) {
  const int part_channels = channels / parts;
  const std::size_t part_elements = static_cast<std::size_t>(part_channels) * height * width;
  const std::size_t full_elements = static_cast<std::size_t>(channels) * height * width;
  std::vector<uint16_t> expected(full_elements);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = FloatToHalfBits(static_cast<float>(static_cast<int>((i * 11 + 1) % 23) - 11));
  }

  mlvc::AclBuffer output_dev(full_elements * sizeof(uint16_t));
  CheckAclStatus(aclrtMemcpy(output_dev.data(), output_dev.bytes(), expected.data(),
                             output_dev.bytes(), ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy restore_y sequence initial output H2D");

  for (int mask = 0; mask < parts; ++mask) {
    const std::vector<int8_t> symbols = MakeSymbols(part_elements);
    std::vector<float> means_source = MakeFullFp32(channels, height, width);
    for (std::size_t i = 0; i < means_source.size(); ++i) {
      means_source[i] += static_cast<float>(mask) / 3.0f;
    }
    const std::vector<uint16_t> means = Fp32ToFp16(means_source);
    mlvc::AclBuffer symbols_dev;
    mlvc::AclBuffer means_dev;
    Upload(symbols, &symbols_dev);
    Upload(means, &means_dev);
    mlvc::DecodePriorRestoreYAcl(static_cast<const int8_t*>(symbols_dev.data()), means_dev.data(),
                                 parts, mask, channels, height, width, output_dev.data(), stream);
    UpdateReferenceRestoreY(symbols, means, parts, mask, channels, height, width, &expected);
  }

  CheckEqualBits(Download<uint16_t>(output_dev, full_elements), expected,
                 parts == 4 ? "restore_y4x_sequence" : "restore_y2x_sequence");
}

void RunApplyQuantCase(int quant_channels, int channels, int height, int width, void* stream) {
  const std::vector<uint16_t> values = Fp32ToFp16(MakeFullFp32(channels, height, width));
  std::vector<float> quant_source(static_cast<std::size_t>(quant_channels) * height * width);
  for (std::size_t i = 0; i < quant_source.size(); ++i) {
    quant_source[i] = 0.25f + static_cast<float>((i * 5 + 2) % 19) / 11.0f;
  }
  const std::vector<uint16_t> quant = Fp32ToFp16(quant_source);

  mlvc::AclBuffer quant_dev;
  mlvc::AclBuffer values_dev;
  mlvc::AclBuffer output_dev(values.size() * sizeof(uint16_t));
  Upload(quant, &quant_dev);
  Upload(values, &values_dev);
  mlvc::DecodePriorApplyChannelQuantStepAcl(quant_dev.data(), quant_channels, values_dev.data(),
                                            channels, height, width, output_dev.data(), stream);
  CheckEqualBits(Download<uint16_t>(output_dev, values.size()),
                 ReferenceApplyQuant(quant, quant_channels, values, channels, height, width),
                 "apply_channel_quant_step");
}

void RunInt8ToFp16Case(std::size_t count, void* stream) {
  const std::vector<int8_t> symbols = MakeSymbols(count);
  mlvc::AclBuffer symbols_dev;
  mlvc::AclBuffer output_dev(count * sizeof(uint16_t));
  Upload(symbols, &symbols_dev);
  const int64_t shape[4] = {1, 1, 1, static_cast<int64_t>(count)};
  mlvc::DecodePriorInt8ToFp16Acl(static_cast<const int8_t*>(symbols_dev.data()), shape, 4, count,
                                 output_dev.data(), stream);
  CheckEqualBits(Download<uint16_t>(output_dev, count), ReferenceInt8ToFp16(symbols),
                 "int8_to_fp16");
}

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  int channels = 16;
  int height = 7;
  int width = 9;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) {
      device = std::stoi(argv[++i]);
    } else if (arg == "--channels" && i + 1 < argc) {
      channels = std::stoi(argv[++i]);
    } else if (arg == "--height" && i + 1 < argc) {
      height = std::stoi(argv[++i]);
    } else if (arg == "--width" && i + 1 < argc) {
      width = std::stoi(argv[++i]);
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }

  try {
    mlvc::Check(channels > 0 && channels % 4 == 0, "channels must be positive and divisible by 4");
    mlvc::Check(height > 0 && width > 0, "height and width must be positive");
    mlvc::AclRuntime runtime(device);
    mlvc::Check(mlvc::DecodePriorAclAvailable(), "decode prior ACLNN custom ops are not available");

    for (int parts : {2, 4}) {
      for (int mask = 0; mask < parts; ++mask) {
        RunSinglePartCase(parts, mask, channels, height, width, runtime.stream());
        RunRestoreYCase(parts, mask, channels, height, width, runtime.stream());
      }
      RunRestoreYSequenceCase(parts, channels, height, width, runtime.stream());
    }
    RunApplyQuantCase(1, channels, height, width, runtime.stream());
    RunApplyQuantCase(channels, channels, height, width, runtime.stream());
    RunInt8ToFp16Case(static_cast<std::size_t>(channels) * height * width + 13, runtime.stream());

    std::cout << "decode_prior_acl channels=" << channels << " height=" << height
              << " width=" << width << " status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "validate_decode_prior_acl failed: " << error.what() << "\n";
    return 1;
  }
}
