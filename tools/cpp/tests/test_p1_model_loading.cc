#include <mlvc/core/status.h>
#include <mlvc/runtime/acl_runtime.h>

#include <functional>
#include <iostream>

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
