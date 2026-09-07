#include <mlvc/core/status.h>
#include <mlvc/runtime/stage_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

struct InputFileBinding {
  std::string input_name;
  std::filesystem::path path;
};

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --manifest <manifest.json> --stage <name> [--device 0]"
               " [--warmup 50] [--iterations 2000]"
               " [--input-file <input_name>=<raw_file>]...\n";
}

bool ParseInputFileBinding(const std::string& value, InputFileBinding* binding) {
  const std::size_t separator = value.find('=');
  if (separator == std::string::npos || separator == 0 || separator + 1 >= value.size()) {
    return false;
  }
  binding->input_name = value.substr(0, separator);
  binding->path = value.substr(separator + 1);
  return true;
}

void LoadInputFiles(const std::vector<InputFileBinding>& bindings,
                    const mlvc::StageModel& stage, std::vector<TensorBuffer>* inputs) {
  for (const InputFileBinding& binding : bindings) {
    std::size_t input_index = stage.record().inputs.size();
    for (std::size_t i = 0; i < stage.record().inputs.size(); ++i) {
      if (stage.record().inputs[i].name == binding.input_name) {
        input_index = i;
        break;
      }
    }
    if (input_index == stage.record().inputs.size()) {
      throw std::invalid_argument("unknown stage input in --input-file: " + binding.input_name);
    }

    std::error_code error;
    const std::uintmax_t file_bytes = std::filesystem::file_size(binding.path, error);
    if (error) {
      throw std::runtime_error("failed to inspect input file " + binding.path.string() +
                               ": " + error.message());
    }
    TensorBuffer& input = inputs->at(input_index);
    if (file_bytes != input.bytes.size()) {
      throw std::invalid_argument("input file size mismatch for " + binding.input_name +
                                  ": expected " + std::to_string(input.bytes.size()) +
                                  " bytes, got " + std::to_string(file_bytes));
    }

    std::ifstream stream(binding.path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(input.bytes.data()),
                static_cast<std::streamsize>(input.bytes.size()));
    if (!stream) {
      throw std::runtime_error("failed to read input file: " + binding.path.string());
    }
    std::cout << "input_file=" << binding.input_name << "=" << binding.path.string()
              << " bytes=" << file_bytes << "\n";
  }
}

TensorBuffer MakeTensor(const mlvc::TensorSpec& spec, uint32_t seed) {
  TensorBuffer buffer;
  buffer.shape = mlvc::TensorShape(spec.shape);
  buffer.dtype = spec.dtype;
  buffer.bytes.resize(buffer.shape.NumElements() * mlvc::ElementSize(spec.dtype));

  if (spec.dtype == mlvc::DataType::kInt32) {
    auto* values = reinterpret_cast<int32_t*>(buffer.bytes.data());
    const std::size_t count = buffer.bytes.size() / sizeof(int32_t);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = static_cast<int32_t>(seed);
    }
    return buffer;
  }

  if (spec.dtype == mlvc::DataType::kFloat16) {
    auto* values = reinterpret_cast<uint16_t*>(buffer.bytes.data());
    const std::size_t count = buffer.bytes.size() / sizeof(uint16_t);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = static_cast<uint16_t>(0x3800);
    }
    return buffer;
  }

  for (std::size_t i = 0; i < buffer.bytes.size(); ++i) {
    buffer.bytes[i] = static_cast<uint8_t>((seed + i * 17U) & 0xffU);
  }
  return buffer;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path manifest_path;
  std::string stage_name;
  int device = 0;
  int warmup = 50;
  int iterations = 2000;
  std::vector<InputFileBinding> input_file_bindings;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--manifest" && i + 1 < argc) {
      manifest_path = argv[++i];
    } else if (arg == "--stage" && i + 1 < argc) {
      stage_name = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::stoi(argv[++i]);
    } else if (arg == "--input-file" && i + 1 < argc) {
      InputFileBinding binding;
      if (!ParseInputFileBinding(argv[++i], &binding)) {
        PrintUsage(argv[0]);
        return 2;
      }
      for (const InputFileBinding& existing : input_file_bindings) {
        if (existing.input_name == binding.input_name) {
          std::cerr << "duplicate --input-file binding: " << binding.input_name << "\n";
          return 2;
        }
      }
      input_file_bindings.push_back(std::move(binding));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }

  if (manifest_path.empty() || stage_name.empty() || warmup < 0 || iterations <= 0) {
    PrintUsage(argv[0]);
    return 2;
  }

  try {
    mlvc::StageRuntime runtime(device);
    mlvc::StageModelSet models(&runtime, mlvc::ModelManifest::Load(manifest_path));
    mlvc::StageModel& stage = models.GetStage(stage_name);

    std::vector<TensorBuffer> inputs;
    std::vector<TensorBuffer> outputs;
    std::vector<mlvc::NamedTensorView> input_views;
    std::vector<mlvc::NamedTensorView> output_views;

    uint32_t seed = 1;
    for (const mlvc::TensorSpec& spec : stage.record().inputs) {
      inputs.push_back(MakeTensor(spec, seed++));
    }
    LoadInputFiles(input_file_bindings, stage, &inputs);
    for (const mlvc::TensorSpec& spec : stage.record().outputs) {
      outputs.push_back(MakeTensor(spec, 0));
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      input_views.push_back({stage.record().inputs[i].name.c_str(), inputs[i].View()});
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      output_views.push_back({stage.record().outputs[i].name.c_str(), outputs[i].View()});
    }

    for (int i = 0; i < warmup; ++i) {
      stage.RunNamed(input_views.data(), input_views.size(), output_views.data(),
                     output_views.size());
    }

    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      stage.RunNamed(input_views.data(), input_views.size(), output_views.data(),
                     output_views.size());
    }
    const auto end = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(end - begin).count();

    std::cout << "stage=" << stage_name << "\n";
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "total_ms=" << total_ms << "\n";
    std::cout << "avg_ms=" << (total_ms / static_cast<double>(iterations)) << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_acl_stage failed: " << error.what() << "\n";
    return 1;
  }
}
