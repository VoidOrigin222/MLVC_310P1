#include <ulbvc/core/status.h>
#include <ulbvc/runtime/acl_runtime.h>

#include <functional>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {
void ExpectRejected(ulbvc::AclRuntime* runtime, const ulbvc::ModelManifest& manifest,
                    ulbvc::ModelRecord record, const std::string& reason) {
  try {
    ulbvc::AclStage stage(runtime, record, manifest.directory() / record.model);
  } catch (const ulbvc::Error& error) {
    ulbvc::Check(std::string(error.what()).find(reason) != std::string::npos,
                "unexpected rejection: " + std::string(error.what()));
    return;
  }
  throw ulbvc::Error("invalid model specification accepted: " + reason);
}

void ExpectRuntimeShapeRejected(ulbvc::AclStage* stage, const ulbvc::TensorSpec& input) {
  const std::size_t element_count = ulbvc::TensorShape(input.shape).NumElements();
  ulbvc::Check(element_count <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
              "test tensor element count exceeds int64 range");
  std::vector<int64_t> wrong_shape = {
      static_cast<int64_t>(element_count)};
  if (wrong_shape == input.shape) {
    wrong_shape = {1, static_cast<int64_t>(element_count)};
  }
  std::vector<uint8_t> storage(element_count * ulbvc::ElementSize(input.dtype));
  ulbvc::NamedTensorView wrong_input{
      input.name.c_str(),
      ulbvc::TensorView(storage.data(), ulbvc::TensorShape(std::move(wrong_shape)), input.dtype,
                       ulbvc::MemoryLocation::kCpu)};
  try {
    stage->RunNamed(&wrong_input, 1, nullptr, 0);
  } catch (const ulbvc::Error& error) {
    ulbvc::Check(std::string(error.what()).find("shape mismatch") != std::string::npos,
                "unexpected runtime rejection: " + std::string(error.what()));
    return;
  }
  throw ulbvc::Error("same-element-count tensor with a different shape was accepted");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    ulbvc::Check(argc == 2, "usage: test_p1_model_loading <manifest>");
    const auto manifest = ulbvc::ModelManifest::Load(argv[1]);
    ulbvc::AclRuntime runtime(0);
    for (const auto& record : manifest.models()) {
      // Exercise the real OM metadata, and retain logical names for RunStage.
      {
        ulbvc::AclStage stage(&runtime, record, manifest.directory() / record.model);
        for (std::size_t i = 0; i < record.outputs.size(); ++i) {
          ulbvc::Check(stage.record().outputs[i].name == record.outputs[i].name,
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
      bad.outputs[0].dtype = record.outputs[0].dtype == ulbvc::DataType::kInt32
                                 ? ulbvc::DataType::kFloat16
                                 : ulbvc::DataType::kInt32;
      ExpectRejected(&runtime, manifest, bad, "dtype mismatch");
      bad = record;
      bad.outputs[0].shape.back() += 1;
      ExpectRejected(&runtime, manifest, bad, "shape mismatch");
      bad = record;
      bad.inputs[0].name = "wrong_input";
      ExpectRejected(&runtime, manifest, bad, "name mismatch");
      bad = record;
      bad.inputs[0].dtype = record.inputs[0].dtype == ulbvc::DataType::kInt32
                                ? ulbvc::DataType::kFloat16
                                : ulbvc::DataType::kInt32;
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
