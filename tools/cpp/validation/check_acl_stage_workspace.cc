#include <ulbvc/codec/detail/stage/stage_runner.h>
#include <ulbvc/codec/detail/stage/stage_runtime_state.h>
#include <ulbvc/codec/detail/stage/stage_workspace.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/status.h>
#include <ulbvc/runtime/stage_runtime.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " --manifest <manifest.json> --stage <name> [--device 0]\n";
}

ulbvc::codec::TensorData MakeTensor(const ulbvc::TensorSpec& spec, uint32_t seed) {
  ulbvc::codec::TensorData tensor;
  tensor.shape = ulbvc::TensorShape(spec.shape);
  tensor.dtype = spec.dtype;
  tensor.bytes.resize(tensor.shape.NumElements() * ulbvc::ElementSize(spec.dtype));
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
    ulbvc::StageRuntime runtime(device);
    ulbvc::StageModelSet models(&runtime, ulbvc::ModelManifest::Load(manifest_path));
    ulbvc::codec::StageOutputWorkspace workspace(&models,
                                                ulbvc::codec::StageOutputBindingMode::kAclDeviceWithCpuMirror);
    ulbvc::codec::g_stage_output_workspace = &workspace;
    ulbvc::codec::g_acl_user_compute_stream = runtime.stream();

    const ulbvc::ModelRecord& record = models.manifest().GetModel(stage_name);
    ulbvc::Check(record.inputs.size() == 1, "check_acl_stage_workspace currently expects one input");
    ulbvc::codec::TensorData input = MakeTensor(record.inputs[0], 1);
    ulbvc::codec::RunOutput output = ulbvc::codec::RunStage(
        &models, stage_name, {{record.inputs[0].name.c_str(), &input, nullptr}}, nullptr);
    ulbvc::Check(output.Size() == record.outputs.size(), "unexpected output count");
    for (std::size_t i = 0; i < output.Size(); ++i) {
      const ulbvc::codec::RunOutput::Entry& entry = output.tensors[i];
      ulbvc::Check(entry.handle != nullptr, "ACL mirror did not attach an output handle");
      ulbvc::Check(entry.handle->acl_valid(), "ACL device output with CPU mirror handle is not ACL-valid");
      ulbvc::Check(entry.handle->cpu_valid(), "ACL device output with CPU mirror handle is not CPU-valid");
      ulbvc::Check(entry.tensor != nullptr && entry.tensor->bytes.size() == entry.handle->bytes(),
                  "ACL mirror CPU tensor was not materialized");
      std::cout << *entry.name << " bytes=" << entry.handle->bytes()
                << " residency=" << entry.handle->ResidencyString() << "\n";
    }
    ulbvc::Check(workspace.CanCopyStageTensorFromDevice(stage_name, record.outputs[0].name),
                "ACL mirror did not expose direct device copy for first output");
    ulbvc::PinnedHostBuffer pinned(output.tensors[0].handle->bytes());
    std::size_t offset = 0;
    ulbvc::codec::TensorBytesView pinned_view;
    workspace.CopyStageTensorFromDevice(stage_name, record.outputs[0].name, &pinned, &offset,
                                        &pinned_view);
    ulbvc::Check(pinned_view.byte_count == output.tensors[0].tensor->bytes.size(),
                "direct ACL-to-pinned copy size mismatch");
    ulbvc::Check(std::memcmp(pinned_view.bytes, output.tensors[0].tensor->bytes.data(),
                            pinned_view.byte_count) == 0,
                "direct ACL-to-pinned copy mismatch");
    std::cout << "direct_device_copy_bytes=" << pinned_view.byte_count << "\n";
    if (record.inputs[0].shape == record.outputs[0].shape &&
        record.inputs[0].dtype == record.outputs[0].dtype) {
      std::optional<ulbvc::codec::InputView> reused =
          workspace.InputViewForCpuMirror(output.tensors[0].tensor);
      ulbvc::Check(reused.has_value(), "ACL mirror did not expose reusable output view");
      ulbvc::Check(reused->device_reuse, "ACL mirror reusable view did not mark device reuse");
      ulbvc::Check(reused->view.location() == ulbvc::MemoryLocation::kAcl,
                  "ACL mirror reusable view is not ACL memory");
      ulbvc::codec::RunOutput second = ulbvc::codec::RunStage(
          &models, stage_name, {{record.inputs[0].name.c_str(), output.tensors[0].tensor, nullptr}},
          nullptr);
      ulbvc::Check(second.Handle(record.outputs[0].name) != nullptr,
                  "second mirror run did not attach output handle");
      std::cout << "reused_input_location=acl\n";
    }

    ulbvc::codec::g_stage_output_workspace = nullptr;
    ulbvc::codec::g_acl_user_compute_stream = nullptr;
    return 0;
  } catch (const std::exception& error) {
    ulbvc::codec::g_stage_output_workspace = nullptr;
    ulbvc::codec::g_acl_user_compute_stream = nullptr;
    std::cerr << "check_acl_stage_workspace failed: " << error.what() << "\n";
    return 1;
  }
}
