#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/status.h>
#include <mlvc/runtime/stage_runtime.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include <mlvc/codec/detail/stage/stage_workspace.h>

namespace {

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " --manifest <manifest.json> --stage <name> [--device 0]\n";
}

mlvc::codec::TensorData MakeTensor(const mlvc::TensorSpec& spec, uint32_t seed) {
  mlvc::codec::TensorData tensor;
  tensor.shape = mlvc::TensorShape(spec.shape);
  tensor.dtype = spec.dtype;
  tensor.bytes.resize(tensor.shape.NumElements() * mlvc::ElementSize(spec.dtype));
  for (std::size_t i = 0; i < tensor.bytes.size(); ++i) {
    tensor.bytes[i] = static_cast<uint8_t>((seed + i * 17U) & 0xffU);
  }
  return tensor;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path manifest_path;
  std::string stage_name;
  int device = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--manifest" && i + 1 < argc) {
      manifest_path = argv[++i];
    } else if (arg == "--stage" && i + 1 < argc) {
      stage_name = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device = std::stoi(argv[++i]);
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }
  if (manifest_path.empty() || stage_name.empty()) {
    PrintUsage(argv[0]);
    return 2;
  }

  try {
    mlvc::StageRuntime runtime(device);
    mlvc::StageModelSet models(&runtime, mlvc::ModelManifest::Load(manifest_path));
    mlvc::codec::StageOutputWorkspace workspace(&models,
                                                mlvc::codec::StageOutputBindingMode::kAclMirror);
    mlvc::codec::g_stage_output_workspace = &workspace;
    mlvc::codec::g_acl_user_compute_stream = runtime.stream();

    const mlvc::ModelRecord& record = models.manifest().GetModel(stage_name);
    mlvc::Check(record.inputs.size() == 1, "check_acl_stage_workspace currently expects one input");
    mlvc::codec::TensorData input = MakeTensor(record.inputs[0], 1);
    mlvc::codec::RunOutput output = mlvc::codec::RunStage(
        &models, stage_name, {{record.inputs[0].name.c_str(), &input, nullptr}}, nullptr);
    mlvc::Check(output.Size() == record.outputs.size(), "unexpected output count");
    for (std::size_t i = 0; i < output.Size(); ++i) {
      const mlvc::codec::RunOutput::Entry& entry = output.tensors[i];
      mlvc::Check(entry.handle != nullptr, "ACL mirror did not attach an output handle");
      mlvc::Check(entry.handle->acl_valid(), "ACL mirror output handle is not ACL-valid");
      mlvc::Check(entry.handle->cpu_valid(), "ACL mirror output handle is not CPU-valid");
      mlvc::Check(entry.tensor != nullptr && entry.tensor->bytes.size() == entry.handle->bytes(),
                  "ACL mirror CPU tensor was not materialized");
      std::cout << *entry.name << " bytes=" << entry.handle->bytes()
                << " residency=" << entry.handle->ResidencyString() << "\n";
    }
    mlvc::Check(workspace.CanCopyStageTensorFromDevice(stage_name, record.outputs[0].name),
                "ACL mirror did not expose direct device copy for first output");
    mlvc::PinnedHostBuffer pinned(output.tensors[0].handle->bytes());
    std::size_t offset = 0;
    mlvc::codec::TensorBytesView pinned_view;
    workspace.CopyStageTensorFromDevice(stage_name, record.outputs[0].name, &pinned, &offset,
                                        &pinned_view);
    mlvc::Check(pinned_view.byte_count == output.tensors[0].tensor->bytes.size(),
                "direct ACL-to-pinned copy size mismatch");
    mlvc::Check(std::memcmp(pinned_view.bytes, output.tensors[0].tensor->bytes.data(),
                            pinned_view.byte_count) == 0,
                "direct ACL-to-pinned copy mismatch");
    std::cout << "direct_device_copy_bytes=" << pinned_view.byte_count << "\n";
    if (record.inputs[0].shape == record.outputs[0].shape &&
        record.inputs[0].dtype == record.outputs[0].dtype) {
      std::optional<mlvc::codec::InputView> reused =
          workspace.InputViewForCpuMirror(output.tensors[0].tensor);
      mlvc::Check(reused.has_value(), "ACL mirror did not expose reusable output view");
      mlvc::Check(reused->device_reuse, "ACL mirror reusable view did not mark device reuse");
      mlvc::Check(reused->view.location() == mlvc::MemoryLocation::kAcl,
                  "ACL mirror reusable view is not ACL memory");
      mlvc::codec::RunOutput second = mlvc::codec::RunStage(
          &models, stage_name, {{record.inputs[0].name.c_str(), output.tensors[0].tensor, nullptr}},
          nullptr);
      mlvc::Check(second.Handle(record.outputs[0].name) != nullptr,
                  "second mirror run did not attach output handle");
      std::cout << "reused_input_location=acl\n";
    }

    mlvc::codec::g_stage_output_workspace = nullptr;
    mlvc::codec::g_acl_user_compute_stream = nullptr;
    return 0;
  } catch (const std::exception& error) {
    mlvc::codec::g_stage_output_workspace = nullptr;
    mlvc::codec::g_acl_user_compute_stream = nullptr;
    std::cerr << "check_acl_stage_workspace failed: " << error.what() << "\n";
    return 1;
  }
}
