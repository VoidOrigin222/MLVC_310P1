#include <acl/acl.h>
#include <ulbvc/codec/detail/profile/codec_profile.h>
#include <ulbvc/codec/detail/stage/constants.h>
#include <ulbvc/codec/detail/stage/stage_runtime_state.h>

#include <chrono>
#include <cstring>
#include <string>

#include "ulbvc/core/status.h"
#include "ulbvc/framework/profile_range.h"

namespace ulbvc::codec {
namespace {

void CheckAclStatus(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw ulbvc::Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

}  // namespace

bool g_skip_async_entropy_cpu_mirror = false;
bool g_skip_async_encode_device_only_cpu_mirror = false;
bool g_skip_async_decode_device_only_cpu_mirror = false;
bool g_force_decode_video_output_cpu_mirror = false;
StageOutputWorkspace* g_stage_output_workspace = nullptr;
void* g_acl_user_compute_stream = nullptr;
ulbvc::CodecGraphExecutor* g_codec_graph_executor = nullptr;
bool g_validate_acl_decode_prior = false;
bool g_enable_stage_fusion = false;

AsyncEntropyInputs CopyEntropyInputsToPinnedFromDevice(
    std::string_view hyper_stage, std::string_view spatial_stage, int part_count,
    ulbvc::PinnedHostBuffer* buffer, float force_zero_thres, PinnedCopySpan* copy_span,
    ulbvc::Profiler* profiler) {
  ULBVC_PROFILE_RANGE_FUNCTION();
  ulbvc::Check(g_stage_output_workspace != nullptr, "stage output workspace is required");
  AsyncEntropyInputs inputs;
  inputs.part_count = part_count;
  std::size_t offset = 0;
  const auto host_begin = std::chrono::steady_clock::now();
  ulbvc::ScopedCodecGraphNode graph_node(
      g_codec_graph_executor, profiler, ulbvc::CodecGraphNodeType::kDeviceCopy,
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
        {ulbvc::Profiler::Arg("hyper_stage", std::string(hyper_stage)),
         ulbvc::Profiler::Arg("spatial_stage", std::string(spatial_stage)),
         ulbvc::Profiler::Arg("direction", "D2H"), ulbvc::Profiler::Arg("reason", "entropy_encode"),
         ulbvc::Profiler::Arg("bytes", static_cast<uint64_t>(offset)),
         ulbvc::Profiler::Arg("part_count", part_count),
         ulbvc::Profiler::BoolArg("y_indexes_prebuilt", y_indexes_prebuilt),
         ulbvc::Profiler::Arg("memory", "device_to_pinned_cpu"),
         ulbvc::Profiler::Arg("residency", "mixed_tensor_handles"),
         ulbvc::Profiler::BoolArg("aggregate", true)});
  }
  return inputs;
}

bool CloneAclHandle(const ulbvc::TensorHandle* source, ulbvc::TensorHandle* destination,
                    std::string_view name, ulbvc::Profiler* profiler) {
  ULBVC_PROFILE_RANGE(std::string("CloneAclHandle.") + std::string(name));
  if (destination != nullptr) {
    *destination = ulbvc::TensorHandle();
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
          {ulbvc::Profiler::Arg("tensor", std::string(name)),
           ulbvc::Profiler::Arg("direction", "D2D"),
           ulbvc::Profiler::Arg("reason", "reference_state_acl_resident_clone"),
           ulbvc::Profiler::Arg("memory", "acl_to_acl"),
           ulbvc::Profiler::Arg("residency", destination->ResidencyString()),
           ulbvc::Profiler::Arg("bytes", static_cast<uint64_t>(source->bytes()))});
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
                            {ulbvc::Profiler::Arg("tensor", std::string(name)),
                             ulbvc::Profiler::Arg("direction", "host_to_host"),
                             ulbvc::Profiler::Arg("reason", "reference_state_clone"),
                             ulbvc::Profiler::Arg("residency", destination->ResidencyString()),
                             ulbvc::Profiler::Arg("bytes", static_cast<uint64_t>(source->bytes()))});
  }
  return true;
}

bool CloneRunOutputHandle(const RunOutput& output, std::string_view tensor_name,
                          ulbvc::TensorHandle* destination, std::string_view name,
                          ulbvc::Profiler* profiler) {
  const ulbvc::TensorHandle* handle = output.Handle(tensor_name);
  return CloneAclHandle(handle, destination, name, profiler);
}

}  // namespace ulbvc::codec
