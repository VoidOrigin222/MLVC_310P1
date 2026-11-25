#include "ulbvc/core/status.h"
#include "ulbvc/runtime/acl_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
struct Options {
  std::filesystem::path manifest, input, expected, output;
  int device = 0;
  double atol = 0.02, rtol = 0.02;
  bool exact = false;
};
Options Parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--exact") { options.exact = true; continue; }
    ulbvc::Check(i + 1 < argc, "missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--manifest") options.manifest = value;
    else if (key == "--input") options.input = value;
    else if (key == "--expected") options.expected = value;
    else if (key == "--output") options.output = value;
    else if (key == "--device") options.device = std::stoi(value);
    else if (key == "--atol") options.atol = std::stod(value);
    else if (key == "--rtol") options.rtol = std::stod(value);
    else throw ulbvc::Error("unknown option " + key);
  }
  ulbvc::Check(!options.manifest.empty() && !options.input.empty() && !options.expected.empty(),
              "usage: check_translation_reference_adaptor --manifest FILE --input FP16 --expected FP16 [--exact] [--output FP16]");
  ulbvc::Check(options.device >= 0 && std::isfinite(options.atol) && std::isfinite(options.rtol) &&
                  options.atol >= 0 && options.rtol >= 0,
              "device and finite tolerances must be nonnegative");
  return options;
}
std::vector<uint8_t> ReadExact(const std::filesystem::path& path, std::size_t size) {
  ulbvc::Check(std::filesystem::file_size(path) == size, "unexpected tensor file byte count: " + path.string());
  std::vector<uint8_t> bytes(size);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  ulbvc::Check(input.good(), "failed to read " + path.string());
  return bytes;
}
double HalfValue(const std::vector<uint8_t>& bytes, std::size_t i) {
  const uint16_t bits = static_cast<uint16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8));
  const int exponent = (bits >> 10) & 31;
  const int fraction = bits & 1023;
  if (exponent == 31) return fraction ? std::numeric_limits<double>::quiet_NaN() :
      ((bits & 0x8000) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity());
  const double magnitude = exponent == 0 ? std::ldexp(static_cast<double>(fraction), -24) :
      std::ldexp(static_cast<double>(1024 + fraction), exponent - 25);
  return (bits & 0x8000) ? -magnitude : magnitude;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = Parse(argc, argv);
    const auto manifest = ulbvc::ModelManifest::Load(options.manifest);
    ulbvc::ValidateAclManifestForAclRuntime(manifest);
    const auto& record = manifest.GetModel("ULBVCReferenceFromFrame");
    ulbvc::Check(record.inputs.size() == 1 && record.outputs.size() == 1,
                "reference adaptor must have exactly one input and output");
    const auto& in = record.inputs[0];
    const auto& out = record.outputs[0];
    ulbvc::Check(in.name == "ref_frame" && out.name == "ref_feature" &&
                    in.dtype == ulbvc::DataType::kFloat16 && out.dtype == ulbvc::DataType::kFloat16 &&
                    in.shape.size() == 4 && out.shape.size() == 4 && in.shape[0] == 1 &&
                    in.shape[1] == 3 && in.shape[2] % 8 == 0 && in.shape[3] % 8 == 0 &&
                    out.shape == std::vector<int64_t>{1, 96, in.shape[2] / 8, in.shape[3] / 8},
                "invalid frame-reference adaptor tensor specification");
    const auto input_count = ulbvc::TensorShape(in.shape).NumElements();
    const auto output_count = ulbvc::TensorShape(out.shape).NumElements();
    auto input = ReadExact(options.input, input_count * 2);
    const auto expected = ReadExact(options.expected, output_count * 2);
    bool constant_half = true;
    for (std::size_t i = 0; i < input_count; ++i) {
      const double value = HalfValue(input, i);
      ulbvc::Check(std::isfinite(value) && value >= 0 && value <= 1, "input reference frame must be finite in [0,1]");
      constant_half = constant_half && value == 0.5;
    }
    const bool exact = options.exact || constant_half;
    const auto model_path = manifest.directory() / record.model;
    ulbvc::Check(std::filesystem::file_size(model_path) == record.bytes, "reference OM byte count disagrees with manifest");
    std::vector<uint8_t> output(output_count * 2);
    ulbvc::AclRuntime runtime(options.device);
    ulbvc::AclStage stage(&runtime, record, model_path);
    stage.Run({{"ref_frame", ulbvc::TensorView(input.data(), ulbvc::TensorShape(in.shape),
                                              ulbvc::DataType::kFloat16, ulbvc::MemoryLocation::kCpu)}},
              {{"ref_feature", ulbvc::TensorView(output.data(), ulbvc::TensorShape(out.shape),
                                                ulbvc::DataType::kFloat16, ulbvc::MemoryLocation::kCpu)}});
    double max_abs = 0, total_abs = 0, feature_max_abs = 0, memory_max_abs = 0;
    std::size_t failures = 0;
    for (std::size_t i = 0; i < output_count; ++i) {
      const double reference = HalfValue(expected, i), actual = HalfValue(output, i);
      ulbvc::Check(std::isfinite(reference) && std::isfinite(actual), "non-finite reference adaptor output");
      const double error = std::abs(actual - reference);
      max_abs = std::max(max_abs, error);
      total_abs += error;
      auto& channel_max = i < output_count / 2 ? feature_max_abs : memory_max_abs;
      channel_max = std::max(channel_max, error);
      if (error > (exact ? 0 : options.atol + options.rtol * std::abs(reference))) ++failures;
    }
    if (!options.output.empty()) {
      ulbvc::Check(!std::filesystem::exists(options.output), "output fixture already exists");
      std::ofstream file(options.output, std::ios::binary);
      file.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size()));
      ulbvc::Check(file.good(), "failed to write actual adaptor output");
    }
    std::cout << std::setprecision(12) << "{\"stage\":\"ULBVCReferenceFromFrame\",\"elements\":" << output_count
              << ",\"shape\":[1,96," << out.shape[2] << ',' << out.shape[3] << "]"
              << ",\"exact_required\":" << (exact ? "true" : "false") << ",\"max_abs\":" << max_abs
              << ",\"mean_abs\":" << total_abs / output_count << ",\"feature_max_abs\":" << feature_max_abs
              << ",\"memory_max_abs\":" << memory_max_abs << ",\"tolerance_failures\":" << failures
              << ",\"status\":\"" << (failures ? "failed" : "passed") << "\"}\n";
    ulbvc::Check(failures == 0, "reference adaptor differs from verified Python FP16 output");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "check_translation_reference_adaptor failed: " << error.what() << '\n';
    return 1;
  }
}
