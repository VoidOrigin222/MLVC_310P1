#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "mlvc/core/status.h"
#include "mlvc/runtime/model_manifest.h"

namespace {

void WriteText(const std::filesystem::path& path, const std::string& text) {
  std::ofstream output(path);
  mlvc::Check(output.good(), "failed to open temp manifest for writing");
  output << text;
}

void ExpectErrorContains(const std::string& label, const std::string& actual,
                         const std::string& expected) {
  mlvc::Check(actual.find(expected) != std::string::npos, label + " error mismatch: " + actual);
}

std::string ManifestWithModels(const std::string& models) {
  return std::string(R"json({
  "created_at": "2026-05-16T00:00:00Z",
  "runtime": "acl",
  "soc_version": "Ascend310P3",
  "dtype": "fp16",
  "sidecar": {"file": "sidecars.mlvcsc", "bytes": 1, "sha256": "00"},
  "models": [
)json") + models +
         R"json(
  ]
})json";
}

}  // namespace

int main() {
  try {
    const std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / "mlvc_manifest_parser_check";
    std::filesystem::create_directories(temp_dir);

    const std::string model = R"json(
    {
      "name": "i_encoder",
      "file": "om_aoe/i_encoder.sim.om",
      "onnx_file": "onnx_original/i_encoder.sim.onnx",
      "optimized_onnx_file": "onnx_optimized/i_encoder.sim.onnx",
      "atc_om_file": "om_atc/i_encoder.sim.om",
      "backend": "acl",
      "bytes": 1,
      "sha256": "01",
      "frame_type": "I",
      "route": "encode",
      "replaces": ["i_encoder_unfused"],
      "inputs": [
        {"name": "x", "dtype": "float16", "shape": [1, 3, 4, 5]},
        {"name": "idx", "dtype": "int32", "shape": [2]}
      ],
      "outputs": [
        {"name": "y", "dtype": "float16", "shape": [1, 3, 4, 5]},
        {"name": "z_symbols", "dtype": "float16", "shape": [1, 64, 6, 10]},
        {"name": "mask", "dtype": "uint8", "shape": [4]},
        {"name": "combined", "dtype": "int16", "shape": [4]},
        {"name": "scores", "dtype": "float32", "shape": [1]}
      ]
    }
)json";
    const std::filesystem::path valid_path = temp_dir / "valid.json";
    WriteText(valid_path, ManifestWithModels(model));
    const mlvc::ModelManifest valid = mlvc::ModelManifest::Load(valid_path);
    mlvc::Check(valid.runtime() == "acl", "runtime field was not parsed");
    mlvc::Check(valid.soc_version() == "Ascend310P3", "soc_version field was not parsed");
    mlvc::Check(valid.models().size() == 1, "valid manifest model count mismatch");
    const mlvc::ModelRecord& parsed = valid.GetModel("i_encoder");
    mlvc::Check(parsed.model == "om_aoe/i_encoder.sim.om", "model file was not parsed");
    mlvc::Check(parsed.onnx_model == "onnx_original/i_encoder.sim.onnx",
                "onnx_file was not parsed");
    mlvc::Check(parsed.optimized_onnx_model == "onnx_optimized/i_encoder.sim.onnx",
                "optimized_onnx_file was not parsed");
    mlvc::Check(parsed.atc_model == "om_atc/i_encoder.sim.om", "atc_om_file was not parsed");
    mlvc::Check(parsed.replaces.size() == 1 && parsed.replaces[0] == "i_encoder_unfused",
                "replaces was not parsed");
    mlvc::Check(parsed.inputs.size() == 2, "valid manifest inputs were not parsed");
    mlvc::Check(parsed.inputs[0].dtype == mlvc::DataType::kFloat16,
                "float16 input dtype was not parsed");
    mlvc::Check(parsed.inputs[1].dtype == mlvc::DataType::kInt32,
                "int32 input dtype was not parsed");
    mlvc::Check(parsed.outputs.size() == 5, "valid manifest outputs were not parsed");
    mlvc::Check(parsed.outputs[1].dtype == mlvc::DataType::kFloat16,
                "z_symbols output dtype was not parsed");
    mlvc::Check(parsed.outputs[2].dtype == mlvc::DataType::kUInt8,
                "uint8 output dtype was not parsed");
    mlvc::Check(parsed.outputs[3].dtype == mlvc::DataType::kInt16,
                "int16 output dtype was not parsed");
    mlvc::Check(parsed.outputs[4].dtype == mlvc::DataType::kFloat32,
                "float32 output dtype was not parsed");
    mlvc::Check(parsed.outputs[1].shape == std::vector<int64_t>({1, 64, 6, 10}),
                "output shape was not parsed");

    const std::filesystem::path duplicate_path = temp_dir / "duplicate.json";
    WriteText(duplicate_path, ManifestWithModels(model + "," + model));
    try {
      (void)mlvc::ModelManifest::Load(duplicate_path);
      throw mlvc::Error("duplicate manifest was unexpectedly accepted");
    } catch (const mlvc::Error& error) {
      ExpectErrorContains("duplicate manifest", error.what(), "duplicated model name");
    }

    std::cout << "model_manifest_parser status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "check_model_manifest_parser failed: " << error.what() << "\n";
    return 1;
  }
}
