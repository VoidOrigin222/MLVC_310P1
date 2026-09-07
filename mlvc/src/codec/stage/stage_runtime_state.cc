#include <mlvc/codec/detail/stage/stage_runtime_state.h>

#include <acl/acl.h>

#include <chrono>
#include <cstring>
#include <string>

#include <mlvc/codec/detail/profile/codec_profile.h>
#include <mlvc/codec/detail/stage/constants.h>
#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc::codec {
namespace {

void CheckAclStatus(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw mlvc::Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

}  // namespace

bool g_skip_async_entropy_cpu_mirror = false;
bool g_skip_async_encode_device_only_cpu_mirror = false;
bool g_skip_async_decode_device_only_cpu_mirror = false;
StageOutputWorkspace* g_stage_output_workspace = nullptr;
void* g_acl_user_compute_stream = nullptr;
mlvc::CodecGraphExecutor* g_codec_graph_executor = nullptr;
bool g_validate_acl_decode_prior = false;
bool g_enable_stage_fusion = false;

AsyncEntropyInputs CopyEntropyInputsToPinnedFromDevice(
    std::string_view hyper_stage, std::string_view spatial_stage, int part_count,
    mlvc::PinnedHostBuffer* buffer, float force_zero_thres, PinnedCopySpan* copy_span,
    mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(g_stage_output_workspace != nullptr, "stage output workspace is required");
  AsyncEntropyInputs inputs;
  inputs.part_count = part_count;
  std::size_t offset = 0;
  const auto host_begin = std::chrono::steady_clock::now();
  mlvc::ScopedCodecGraphNode graph_node(
      g_codec_graph_executor, profiler, mlvc::CodecGraphNodeType::kDeviceCopy,
      std::string(hyper_stage) + "_entropy_inputs", "CopyEntropyInputsToPinnedFromDevice");
  g_stage_output_workspace->CopyStageTensorFromDevice(hyper_stage, "z_symbols", buffer, &offset,
                                                      &inputs.z_symbols, profiler);
  const std::size_t y_payload_offset = offset;
  bool y_indexes_prebuilt = true;
  for (int part = 0; part < part_count; ++part) {
    if (!g_stage_output_workspace->BuildEncodeYIndexesToPinned(
            spatial_stage, part, force_zero_thres, buffer, &offset,
            &inputs.y_combined_indexes[part], &inputs.y_keep_masks[part], profiler)) {
      y_indexes_prebuilt = false;
      break;
    }
  }
  if (!y_indexes_prebuilt) {
    offset = y_payload_offset;
    for (int part = 0; part < part_count; ++part) {
      g_stage_output_workspace->CopyStageTensorFromDevice(spatial_stage, kYResidualPartNames[part],
                                                          buffer, &offset, &inputs.y_symbols[part],
                                                          profiler);
      g_stage_output_workspace->CopyStageTensorFromDevice(
          spatial_stage, kScalePartNames[part], buffer, &offset, &inputs.scales[part], profiler);
    }
  }
  inputs.y_indexes_prebuilt = y_indexes_prebuilt;
  g_stage_output_workspace->SynchronizeCopyStream(profiler, "entropy_encode");
  const auto host_end = std::chrono::steady_clock::now();
  if (profiler != nullptr) {
    copy_span->start_ms = profiler->StartMs(host_begin);
    copy_span->duration_ms = profiler->DurationMs(host_begin, host_end);
    AddProfileEventWithArgs(
        profiler, "copy.d2h.entropy_inputs." + std::string(hyper_stage), "copy", "copy",
        copy_span->start_ms, copy_span->duration_ms,
        {mlvc::Profiler::Arg("hyper_stage", std::string(hyper_stage)),
         mlvc::Profiler::Arg("spatial_stage", std::string(spatial_stage)),
         mlvc::Profiler::Arg("direction", "D2H"), mlvc::Profiler::Arg("reason", "entropy_encode"),
         mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(offset)),
         mlvc::Profiler::Arg("part_count", part_count),
         mlvc::Profiler::BoolArg("y_indexes_prebuilt", y_indexes_prebuilt),
         mlvc::Profiler::Arg("memory", "device_to_pinned_cpu"),
         mlvc::Profiler::Arg("residency", "mixed_tensor_handles"),
         mlvc::Profiler::BoolArg("aggregate", true)});
  }
  return inputs;
}

bool CloneAclHandle(const mlvc::TensorHandle* source, mlvc::TensorHandle* destination,
                    std::string_view name, mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE(std::string("CloneAclHandle.") + std::string(name));
  if (destination != nullptr) {
    *destination = mlvc::TensorHandle();
  }
  if (source == nullptr || destination == nullptr) {
    return false;
  }
  if (source->has_acl_buffer() && source->acl_valid()) {
    destination->Reset(source->shape(), source->dtype());
    destination->UseOwnedCpuBuffer(false);
    destination->AllocateAclBuffer(false);
    if (source->has_cpu_buffer() && source->cpu_valid()) {
      std::memcpy(destination->CpuView().data(), source->CpuView().data(), source->bytes());
      destination->MarkCpuValid();
    }
    const auto begin = std::chrono::steady_clock::now();
    if (g_acl_user_compute_stream != nullptr) {
      source->WaitReady(g_acl_user_compute_stream);
    }
    CheckAclStatus(
        aclrtMemcpy(destination->AclView().data(), destination->bytes(), source->AclView().data(),
                    source->bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE),
        "aclrtMemcpy reference state ACL to ACL");
    destination->MarkAclValid();
    if (g_acl_user_compute_stream != nullptr) {
      destination->RecordReady(g_acl_user_compute_stream);
    }
    const auto end = std::chrono::steady_clock::now();
    if (profiler != nullptr) {
      AddProfileEventWithArgs(
          profiler, "copy.d2d.reference_state." + std::string(name), "copy", "copy",
          profiler->StartMs(begin), profiler->DurationMs(begin, end),
          {mlvc::Profiler::Arg("tensor", std::string(name)),
           mlvc::Profiler::Arg("direction", "D2D"),
           mlvc::Profiler::Arg("reason", "reference_state_acl_resident_clone"),
           mlvc::Profiler::Arg("memory", "acl_to_acl"),
           mlvc::Profiler::Arg("residency", destination->ResidencyString()),
           mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(source->bytes()))});
    }
    return true;
  }
  if (!source->has_cpu_buffer() || !source->cpu_valid()) {
    return false;
  }
  destination->Reset(source->shape(), source->dtype());
  destination->UseOwnedCpuBuffer(false);
  const auto host_begin = std::chrono::steady_clock::now();
  std::memcpy(destination->CpuView().data(), source->CpuView().data(), source->bytes());
  destination->MarkCpuValid();
  const auto host_end = std::chrono::steady_clock::now();
  if (profiler != nullptr) {
    AddProfileEventWithArgs(profiler, "copy.host.reference_state." + std::string(name), "copy",
                            "copy", profiler->StartMs(host_begin),
                            profiler->DurationMs(host_begin, host_end),
                            {mlvc::Profiler::Arg("tensor", std::string(name)),
                             mlvc::Profiler::Arg("direction", "host_to_host"),
                             mlvc::Profiler::Arg("reason", "reference_state_clone"),
                             mlvc::Profiler::Arg("residency", destination->ResidencyString()),
                             mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(source->bytes()))});
  }
  return true;
}

bool CloneRunOutputHandle(const RunOutput& output, std::string_view tensor_name,
                          mlvc::TensorHandle* destination, std::string_view name,
                          mlvc::Profiler* profiler) {
  const mlvc::TensorHandle* handle = output.Handle(tensor_name);
  return CloneAclHandle(handle, destination, name, profiler);
}

}  // namespace mlvc::codec
