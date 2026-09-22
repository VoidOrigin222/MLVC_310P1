#include <mlvc/core/status.h>
#include <mlvc/runtime/acl_runtime.h>

#include <functional>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {
void ExpectRejected(mlvc::AclRuntime* runtime, const mlvc::ModelManifest& manifest,
                    mlvc::ModelRecord record, const std::string& reason) {
  try {
    mlvc::AclStage stage(runtime, record, manifest.directory() / record.model);
  } catch (const mlvc::Error& error) {
    mlvc::Check(std::string(error.what()).find(reason) != std::string::npos,
                "unexpected rejection: " + std::string(error.what()));
    return;
  }
  throw mlvc::Error("invalid model specification accepted: " + reason);
}

void ExpectRuntimeShapeRejected(mlvc::AclStage* stage, const mlvc::TensorSpec& input) {
  const std::size_t element_count = mlvc::TensorShape(input.shape).NumElements();
  mlvc::Check(element_count <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
              "test tensor element count exceeds int64 range");
  std::vector<int64_t> wrong_shape = {
      static_cast<int64_t>(element_count)};
  if (wrong_shape == input.shape) {
    wrong_shape = {1, static_cast<int64_t>(element_count)};
  }
  std::vector<uint8_t> storage(element_count * mlvc::ElementSize(input.dtype));
  mlvc::NamedTensorView wrong_input{
      input.name.c_str(),
      mlvc::TensorView(storage.data(), mlvc::TensorShape(std::move(wrong_shape)), input.dtype,
                       mlvc::MemoryLocation::kCpu)};
  try {
    stage->RunNamed(&wrong_input, 1, nullptr, 0);
  } catch (const mlvc::Error& error) {
    mlvc::Check(std::string(error.what()).find("shape mismatch") != std::string::npos,
                "unexpected runtime rejection: " + std::string(error.what()));
    return;
  }
  throw mlvc::Error("same-element-count tensor with a different shape was accepted");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    mlvc::Check(argc == 2, "usage: test_p1_model_loading <manifest>");
    const auto manifest = mlvc::ModelManifest::Load(argv[1]);
    mlvc::AclRuntime runtime(0);
    for (const auto& record : manifest.models()) {
      // Exercise the real OM metadata, and retain logical names for RunStage.
      {
        mlvc::AclStage stage(&runtime, record, manifest.directory() / record.model);
        for (std::size_t i = 0; i < record.outputs.size(); ++i) {
          mlvc::Check(stage.record().outputs[i].name == record.outputs[i].name,
                      "logical output name changed");
        }
        ExpectRuntimeShapeRejected(&stage, record.inputs[0]);
      }
      auto bad = record;
      bad.outputs[0].name = "wrong_feature";
      ExpectRejected(&runtime, manifest, bad, "name mismatch");
      bad = record;
      bad.outputs[0].name = "ature";  // A substring is not an ATC alias.
      ExpectRejected(&runtime, manifest, bad, "name mismatch");
      bad = record;
      bad.outputs[0].dtype = record.outputs[0].dtype == mlvc::DataType::kInt32
                                 ? mlvc::DataType::kFloat16
                                 : mlvc::DataType::kInt32;
      ExpectRejected(&runtime, manifest, bad, "dtype mismatch");
      bad = record;
      bad.outputs[0].shape.back() += 1;
      ExpectRejected(&runtime, manifest, bad, "shape mismatch");
      bad = record;
      bad.inputs[0].name = "wrong_input";
      ExpectRejected(&runtime, manifest, bad, "name mismatch");
      bad = record;
      bad.inputs[0].dtype = record.inputs[0].dtype == mlvc::DataType::kInt32
                                ? mlvc::DataType::kFloat16
                                : mlvc::DataType::kInt32;
      ExpectRejected(&runtime, manifest, bad, "dtype mismatch");
      bad = record;
      bad.inputs[0].shape.back() += 1;
      ExpectRejected(&runtime, manifest, bad, "shape mismatch");
      bad = record;
      bad.outputs.push_back(bad.outputs.back());
      ExpectRejected(&runtime, manifest, bad, "output count");
      bad = record;
      bad.inputs.push_back(bad.inputs.back());
      ExpectRejected(&runtime, manifest, bad, "input count");
      std::cout << record.name << " positive/negative cases passed\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
