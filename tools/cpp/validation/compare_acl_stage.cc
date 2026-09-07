#include <mlvc/core/status.h>
#include <mlvc/runtime/stage_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct TensorBuffer {
  mlvc::TensorShape shape;
  mlvc::DataType dtype = mlvc::DataType::kFloat16;
  std::vector<uint8_t> bytes;

  mlvc::TensorView View() {
    return mlvc::TensorView::Borrowed(bytes.data(), &shape, dtype, mlvc::MemoryLocation::kCpu);
  }
};

struct OutputDiff {
  std::string name;
  std::size_t elements = 0;
  std::size_t mismatches = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;
};

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --manifest <manifest.json> --stage <name> [--device 0]"
               " [--baseline-om <path>] [--target-om <path>] [--atol 0.005] [--rtol 0.005]\n";
}

uint32_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000U;
  const uint32_t float_exponent = (bits >> 23) & 0xffU;
  uint32_t mantissa = bits & 0x7fffffU;
  if (float_exponent == 0xffU) {
    return sign | (mantissa == 0 ? 0x7c00U : 0x7e00U);
  }
  int32_t exponent = static_cast<int32_t>(float_exponent) - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000U;
    const int shift = 14 - exponent;
    uint32_t half_mantissa = mantissa >> shift;
    const uint32_t remainder = mantissa & ((uint32_t{1} << shift) - 1U);
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0U)) {
      ++half_mantissa;
    }
    return sign | half_mantissa;
  }
  if (exponent >= 31) {
    return sign | 0x7c00U;
  }
  uint32_t half_mantissa = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0U)) {
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
  const uint32_t sign = (value & 0x8000U) << 16;
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

TensorBuffer MakeInput(const mlvc::TensorSpec& spec, uint32_t seed) {
  TensorBuffer buffer;
  buffer.shape = mlvc::TensorShape(spec.shape);
  buffer.dtype = spec.dtype;
  const std::size_t elements = buffer.shape.NumElements();
  buffer.bytes.resize(elements * mlvc::ElementSize(spec.dtype));
  switch (spec.dtype) {
    case mlvc::DataType::kFloat16: {
      auto* output = reinterpret_cast<uint16_t*>(buffer.bytes.data());
      for (std::size_t i = 0; i < elements; ++i) {
        const float value = (static_cast<int>((i + seed) % 23) - 11) / 16.0f;
        output[i] = static_cast<uint16_t>(FloatToHalfBits(value));
      }
      break;
    }
    case mlvc::DataType::kFloat32: {
      auto* output = reinterpret_cast<float*>(buffer.bytes.data());
      for (std::size_t i = 0; i < elements; ++i) {
        output[i] = (static_cast<int>((i + seed) % 23) - 11) / 16.0f;
      }
      break;
    }
    case mlvc::DataType::kInt8: {
      auto* output = reinterpret_cast<int8_t*>(buffer.bytes.data());
      for (std::size_t i = 0; i < elements; ++i) {
        output[i] = static_cast<int8_t>(static_cast<int>((i + seed) % 17) - 8);
      }
      break;
    }
    case mlvc::DataType::kInt16: {
      auto* output = reinterpret_cast<int16_t*>(buffer.bytes.data());
      for (std::size_t i = 0; i < elements; ++i) {
        output[i] = static_cast<int16_t>(static_cast<int>((i + seed) % 257) - 128);
      }
      break;
    }
    case mlvc::DataType::kInt32: {
      auto* output = reinterpret_cast<int32_t*>(buffer.bytes.data());
      for (std::size_t i = 0; i < elements; ++i) {
        output[i] = static_cast<int32_t>(static_cast<int>((i + seed) % 257) - 128);
      }
      break;
    }
    case mlvc::DataType::kUInt8: {
      for (std::size_t i = 0; i < elements; ++i) {
        buffer.bytes[i] = static_cast<uint8_t>((i + seed) % 251);
      }
      break;
    }
  }
  return buffer;
}

TensorBuffer MakeOutput(const mlvc::TensorSpec& spec) {
  TensorBuffer buffer;
  buffer.shape = mlvc::TensorShape(spec.shape);
  buffer.dtype = spec.dtype;
  buffer.bytes.resize(buffer.shape.NumElements() * mlvc::ElementSize(spec.dtype));
  return buffer;
}

double ReadValue(const TensorBuffer& buffer, std::size_t index) {
  switch (buffer.dtype) {
    case mlvc::DataType::kFloat16:
      return HalfBitsToFloat(reinterpret_cast<const uint16_t*>(buffer.bytes.data())[index]);
    case mlvc::DataType::kFloat32:
      return reinterpret_cast<const float*>(buffer.bytes.data())[index];
    case mlvc::DataType::kInt8:
      return reinterpret_cast<const int8_t*>(buffer.bytes.data())[index];
    case mlvc::DataType::kInt16:
      return reinterpret_cast<const int16_t*>(buffer.bytes.data())[index];
    case mlvc::DataType::kInt32:
      return reinterpret_cast<const int32_t*>(buffer.bytes.data())[index];
    case mlvc::DataType::kUInt8:
      return reinterpret_cast<const uint8_t*>(buffer.bytes.data())[index];
  }
  throw mlvc::Error("unsupported dtype");
}

OutputDiff CompareOutput(const std::string& name, const TensorBuffer& baseline,
                         const TensorBuffer& target, double atol, double rtol) {
  mlvc::Check(baseline.dtype == target.dtype, "output dtype mismatch: " + name);
  mlvc::Check(baseline.shape.dims() == target.shape.dims(), "output shape mismatch: " + name);
  OutputDiff diff;
  diff.name = name;
  diff.elements = baseline.shape.NumElements();
  for (std::size_t i = 0; i < diff.elements; ++i) {
    const double expected = ReadValue(baseline, i);
    const double actual = ReadValue(target, i);
    const double abs_error = std::abs(actual - expected);
    const double rel_error = abs_error / std::max(std::abs(expected), 1.0);
    diff.max_abs = std::max(diff.max_abs, abs_error);
    diff.max_rel = std::max(diff.max_rel, rel_error);
    if (!std::isfinite(abs_error) || abs_error > atol + rtol * std::abs(expected)) {
      ++diff.mismatches;
    }
  }
  return diff;
}

std::filesystem::path ResolvePath(const std::filesystem::path& base,
                                  const std::filesystem::path& path) {
  return path.is_absolute() ? path : base / path;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path manifest_path;
  std::filesystem::path baseline_om;
  std::filesystem::path target_om;
  std::string stage_name;
  int device = 0;
  double atol = 0.005;
  double rtol = 0.005;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--manifest" && i + 1 < argc) {
      manifest_path = argv[++i];
    } else if (arg == "--stage" && i + 1 < argc) {
      stage_name = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device = std::stoi(argv[++i]);
    } else if (arg == "--baseline-om" && i + 1 < argc) {
      baseline_om = argv[++i];
    } else if (arg == "--target-om" && i + 1 < argc) {
      target_om = argv[++i];
    } else if (arg == "--atol" && i + 1 < argc) {
      atol = std::stod(argv[++i]);
    } else if (arg == "--rtol" && i + 1 < argc) {
      rtol = std::stod(argv[++i]);
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }

  if (manifest_path.empty() || stage_name.empty() || atol < 0.0 || rtol < 0.0) {
    PrintUsage(argv[0]);
    return 2;
  }

  try {
    mlvc::StageRuntime runtime(device);
    mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(manifest_path);
    const mlvc::ModelRecord& record = manifest.GetModel(stage_name);
    if (baseline_om.empty()) {
      mlvc::Check(!record.atc_model.empty(), "manifest stage has no atc_om_file: " + stage_name);
      baseline_om = ResolvePath(manifest.directory(), record.atc_model);
    }
    if (target_om.empty()) {
      target_om = ResolvePath(manifest.directory(), record.model);
    }
    baseline_om = ResolvePath(manifest.directory(), baseline_om);
    target_om = ResolvePath(manifest.directory(), target_om);

    mlvc::AclStage baseline(&runtime, record, baseline_om);
    mlvc::AclStage target(&runtime, record, target_om);

    std::vector<TensorBuffer> inputs;
    std::vector<TensorBuffer> baseline_outputs;
    std::vector<TensorBuffer> target_outputs;
    std::vector<mlvc::NamedTensorView> input_views;
    std::vector<mlvc::NamedTensorView> baseline_output_views;
    std::vector<mlvc::NamedTensorView> target_output_views;

    uint32_t seed = 1;
    for (const mlvc::TensorSpec& spec : record.inputs) {
      inputs.push_back(MakeInput(spec, seed++));
    }
    for (const mlvc::TensorSpec& spec : record.outputs) {
      baseline_outputs.push_back(MakeOutput(spec));
      target_outputs.push_back(MakeOutput(spec));
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      input_views.push_back({record.inputs[i].name.c_str(), inputs[i].View()});
    }
    for (std::size_t i = 0; i < record.outputs.size(); ++i) {
      baseline_output_views.push_back({record.outputs[i].name.c_str(), baseline_outputs[i].View()});
      target_output_views.push_back({record.outputs[i].name.c_str(), target_outputs[i].View()});
    }

    baseline.RunNamed(input_views.data(), input_views.size(), baseline_output_views.data(),
                      baseline_output_views.size());
    target.RunNamed(input_views.data(), input_views.size(), target_output_views.data(),
                    target_output_views.size());

    std::size_t total_mismatches = 0;
    std::cout << "stage=" << stage_name << "\n";
    std::cout << "baseline_om=" << baseline_om << "\n";
    std::cout << "target_om=" << target_om << "\n";
    std::cout << "atol=" << atol << "\n";
    std::cout << "rtol=" << rtol << "\n";
    for (std::size_t i = 0; i < record.outputs.size(); ++i) {
      const OutputDiff diff =
          CompareOutput(record.outputs[i].name, baseline_outputs[i], target_outputs[i], atol, rtol);
      total_mismatches += diff.mismatches;
      std::cout << "output=" << diff.name << " elements=" << diff.elements
                << " mismatches=" << diff.mismatches << " max_abs=" << diff.max_abs
                << " max_rel=" << diff.max_rel << "\n";
    }
    if (total_mismatches != 0) {
      std::cerr << "compare_acl_stage failed: total_mismatches=" << total_mismatches << "\n";
      return 1;
    }
    std::cout << "status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "compare_acl_stage failed: " << error.what() << "\n";
    return 1;
  }
}
