#include <mlvc/codec/detail/stage/stage_workspace.h>

#include <acl/acl.h>

#include <chrono>
#include <cstring>

#include <mlvc/codec/detail/profile/codec_profile.h>
#include <mlvc/codec/detail/stage/stage_output_policy.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include <mlvc/codec/detail/tensor/tensor_utils.h>
#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"
#include "mlvc/runtime/decode_prior_acl.h"
#include "mlvc/runtime/encode_index_acl.h"

namespace mlvc::codec {
namespace {

void CheckAclStatus(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw mlvc::Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

std::size_t FindOutputIndex(const StageOutputWorkspace::Entry& entry,
                            std::string_view tensor_name) {
  for (std::size_t i = 0; i < entry.size; ++i) {
    if (entry.names[i] != nullptr && *entry.names[i] == tensor_name) {
      return i;
    }
  }
  throw mlvc::Error("missing stage output tensor: " + entry.stage_name + "." +
                    std::string(tensor_name));
}

std::size_t FindInputIndex(const StageOutputWorkspace::Entry& entry, std::string_view tensor_name) {
  for (std::size_t i = 0; i < entry.input_size; ++i) {
    if (entry.input_names[i] != nullptr && *entry.input_names[i] == tensor_name) {
      return i;
    }
  }
  throw mlvc::Error("missing stage input tensor: " + entry.stage_name + "." +
                    std::string(tensor_name));
}

void CopyAclTensorToPinned(const mlvc::TensorHandle& handle, mlvc::PinnedHostBuffer* buffer,
                           std::size_t* offset, TensorBytesView* view) {
  mlvc::Check(buffer != nullptr && offset != nullptr && view != nullptr,
              "ACL pinned copy arguments must not be null");
  mlvc::Check(handle.has_acl_buffer() && handle.acl_valid(), "stage tensor is not ACL-resident");
  const std::size_t next_offset = *offset + handle.bytes();
  mlvc::Check(next_offset <= buffer->bytes(), "async pinned entropy slot is too small");
  auto* destination = static_cast<uint8_t*>(buffer->data()) + *offset;
  CheckAclStatus(aclrtMemcpy(destination, handle.bytes(), handle.AclView().data(), handle.bytes(),
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy stage tensor ACL to pinned CPU");
  *view = TensorBytesView{handle.shape(), handle.dtype(), destination, handle.bytes()};
  *offset = next_offset;
}

}  // namespace

StageOutputWorkspace::StageOutputWorkspace(mlvc::StageModelSet* models,
                                           StageOutputBindingMode binding_mode)
    : binding_mode_(binding_mode) {
  entries_.reserve(models->manifest().models().size());
  for (const mlvc::ModelRecord& record : models->manifest().models()) {
    mlvc::Check(record.outputs.size() <= kMaxStageOutputs,
                "stage output count exceeds fixed storage");
    Entry entry;
    entry.stage_name = record.name;
    entry.input_size = record.inputs.size();
    entry.size = record.outputs.size();
    mlvc::Check(record.inputs.size() <= kMaxStageInputs, "stage input count exceeds fixed storage");
    for (std::size_t i = 0; i < record.inputs.size(); ++i) {
      entry.input_names[i] = &record.inputs[i].name;
      if (binding_mode_ == StageOutputBindingMode::kAclMirror) {
        entry.input_handles[i].Reset(mlvc::TensorShape(record.inputs[i].shape),
                                     record.inputs[i].dtype);
        entry.input_handles[i].UseOwnedCpuBuffer(false);
        entry.input_handles[i].AllocateAclBuffer(false);
      }
    }
    for (std::size_t i = 0; i < record.outputs.size(); ++i) {
      entry.names[i] = &record.outputs[i].name;
      entry.tensors[i] = MakeTensorLike(record.outputs[i]);
      if (binding_mode_ == StageOutputBindingMode::kAclMirror) {
        entry.handles[i].Reset(entry.tensors[i].shape, entry.tensors[i].dtype);
        entry.handles[i].UseOwnedCpuBuffer(false);
        entry.handles[i].AllocateAclBuffer(false);
      }
    }
    entries_.push_back(std::move(entry));
  }
}

RunOutput StageOutputWorkspace::Bind(const mlvc::ModelRecord& record) {
  Entry* entry = Find(record.name);
  mlvc::Check(entry->size == record.outputs.size(), "stage workspace output count mismatch");
  RunOutput output;
  output.size = entry->size;
  for (std::size_t i = 0; i < entry->size; ++i) {
    output.tensors[i] = RunOutput::Entry{
        entry->names[i], &entry->tensors[i],
        binding_mode_ == StageOutputBindingMode::kAclMirror ? &entry->handles[i] : nullptr};
  }
  return output;
}

void StageOutputWorkspace::BindOutputViews(
    const mlvc::ModelRecord& record, const RunOutput& output,
    std::array<mlvc::NamedTensorView, kMaxStageOutputs>* output_views, std::size_t* output_count) {
  Entry* entry = Find(record.name);
  mlvc::Check(entry->size == output.size, "stage workspace output count mismatch");
  *output_count = 0;
  for (std::size_t i = 0; i < output.size; ++i) {
    const mlvc::TensorSpec& spec = record.outputs[i];
    if (binding_mode_ == StageOutputBindingMode::kAclMirror) {
      mlvc::Check(output.tensors[i].handle != nullptr,
                  "ACL mirror output handle is required: " + record.name + "." + spec.name);
      (*output_views)[(*output_count)++] =
          mlvc::NamedTensorView{spec.name.c_str(), output.tensors[i].handle->AclView()};
    } else {
      (*output_views)[(*output_count)++] =
          mlvc::NamedTensorView{spec.name.c_str(), output.tensors[i].tensor->View()};
    }
  }
}

void StageOutputWorkspace::MirrorOutputsToCpu(const mlvc::ModelRecord& record,
                                              const RunOutput& output, mlvc::Profiler* profiler) {
  if (binding_mode_ != StageOutputBindingMode::kAclMirror) {
    return;
  }
  mlvc::Check(output.size == record.outputs.size(), "stage workspace output count mismatch");
  for (std::size_t i = 0; i < output.size; ++i) {
    mlvc::TensorHandle* handle = output.tensors[i].handle;
    TensorData* tensor = output.tensors[i].tensor;
    mlvc::Check(handle != nullptr && tensor != nullptr,
                "ACL mirror output requires both TensorHandle and TensorData");
    handle->MarkAclModified();
    const MaterializationReason reason =
        StageOutputMaterializationReason(record.name, record.outputs[i].name);
    const char* skip_reason = StageOutputCpuMirrorSkipReason(
        record.name, record.outputs[i], reason, g_skip_async_entropy_cpu_mirror,
        g_skip_async_encode_device_only_cpu_mirror, g_skip_async_decode_device_only_cpu_mirror,
        mlvc::DecodePriorAclAvailable());
    if (skip_reason != nullptr) {
      if (profiler != nullptr) {
        AddProfileEventWithArgs(
            profiler, "copy.d2h.skip." + record.name + "." + record.outputs[i].name, "copy", "copy",
            0.0, 0.0,
            {mlvc::Profiler::Arg("stage", record.name),
             mlvc::Profiler::Arg("tensor", record.outputs[i].name),
             mlvc::Profiler::Arg("direction", "D2H"),
             mlvc::Profiler::Arg("reason", MaterializationReasonName(reason)),
             mlvc::Profiler::Arg("skip_reason", skip_reason),
             mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(handle->bytes())),
             mlvc::Profiler::Arg("residency", handle->ResidencyString())});
      }
      continue;
    }
    const auto begin = std::chrono::steady_clock::now();
    const mlvc::TensorCopyResult copy =
        handle->MaterializeToCpu(MaterializationReasonName(reason), nullptr);
    const auto end = std::chrono::steady_clock::now();
    tensor->shape = handle->shape();
    tensor->dtype = handle->dtype();
    if (tensor->bytes.data() != handle->CpuView().data()) {
      tensor->bytes.resize(handle->bytes());
      if (handle->bytes() != 0) {
        std::memcpy(tensor->bytes.data(), handle->CpuView().data(), handle->bytes());
      }
    }
    if (profiler != nullptr && copy.copied) {
      AddProfileEventWithArgs(
          profiler, "copy.d2h.stage_output." + record.name + "." + record.outputs[i].name, "copy",
          "copy", profiler->StartMs(begin), profiler->DurationMs(begin, end),
          {mlvc::Profiler::Arg("stage", record.name),
           mlvc::Profiler::Arg("tensor", record.outputs[i].name),
           mlvc::Profiler::Arg("reason", MaterializationReasonName(reason)),
           mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(copy.bytes)),
           mlvc::Profiler::Arg("memory", "acl_to_cpu"),
           mlvc::Profiler::Arg("residency", handle->ResidencyString())});
    }
  }
}

bool StageOutputWorkspace::CanCopyStageTensorFromDevice(std::string_view stage_name,
                                                        std::string_view tensor_name) const {
  if (binding_mode_ != StageOutputBindingMode::kAclMirror) {
    return false;
  }
  const Entry* entry = FindOrNull(stage_name);
  if (entry == nullptr) {
    return false;
  }
  for (std::size_t i = 0; i < entry->size; ++i) {
    if (entry->names[i] != nullptr && *entry->names[i] == tensor_name &&
        entry->handles[i].has_acl_buffer() && entry->handles[i].acl_valid()) {
      return true;
    }
  }
  return false;
}

std::optional<InputView> StageOutputWorkspace::InputViewForCpuMirror(const TensorData* tensor) {
  if (binding_mode_ != StageOutputBindingMode::kAclMirror) {
    return std::nullopt;
  }
  if (tensor == nullptr) {
    return std::nullopt;
  }
  for (Entry& entry : entries_) {
    for (std::size_t i = 0; i < entry.size; ++i) {
      if (&entry.tensors[i] == tensor && entry.handles[i].acl_valid()) {
        return InputView{entry.handles[i].AclView(), true};
      }
    }
  }
  return std::nullopt;
}

std::optional<InputView> StageOutputWorkspace::UploadInputToDevice(const mlvc::ModelRecord& record,
                                                                   const char* input_name,
                                                                   const TensorData& tensor,
                                                                   mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE(std::string("UploadInputToDevice.") + record.name + "." + input_name);
  if (binding_mode_ != StageOutputBindingMode::kAclMirror) {
    return std::nullopt;
  }
  mlvc::Check(input_name != nullptr, "ACL input upload requires an input name");
  Entry* entry = Find(record.name);
  const std::size_t index = FindInputIndex(*entry, input_name);
  mlvc::TensorHandle& handle = entry->input_handles[index];
  mlvc::Check(handle.bytes() == tensor.bytes.size(), "ACL input workspace byte size mismatch");
  if (IsQScaleInput(input_name) && entry->input_acl_sources[index] == &tensor &&
      handle.acl_valid()) {
    if (profiler != nullptr) {
      AddProfileEventWithArgs(
          profiler, "copy.h2d.skip." + record.name + "." + input_name, "copy", "copy", 0.0, 0.0,
          {mlvc::Profiler::Arg("stage", record.name), mlvc::Profiler::Arg("tensor", input_name),
           mlvc::Profiler::Arg("direction", "H2D"),
           mlvc::Profiler::Arg("reason", "q_scale_acl_cache"),
           mlvc::Profiler::Arg("skip_reason", "q_scale_acl_cache_hit"),
           mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(tensor.bytes.size())),
           mlvc::Profiler::Arg("residency", handle.ResidencyString())});
    }
    return InputView{handle.AclView(), false};
  }

  const auto begin = std::chrono::steady_clock::now();
  handle.AttachCpuBuffer(const_cast<uint8_t*>(tensor.bytes.data()), tensor.bytes.size(), true);
  handle.MarkCpuModified();
  const mlvc::TensorCopyResult copy = handle.EnsureAcl(nullptr);
  handle.RecordReady(g_acl_user_compute_stream);
  const auto end = std::chrono::steady_clock::now();
  if (IsQScaleInput(input_name)) {
    entry->input_acl_sources[index] = &tensor;
  } else {
    entry->input_acl_sources[index] = nullptr;
  }
  if (profiler != nullptr && copy.copied) {
    AddProfileEventWithArgs(
        profiler, "copy.h2d." + record.name + "." + input_name, "copy", "copy",
        profiler->StartMs(begin), profiler->DurationMs(begin, end),
        {mlvc::Profiler::Arg("stage", record.name), mlvc::Profiler::Arg("tensor", input_name),
         mlvc::Profiler::Arg("direction", "H2D"), mlvc::Profiler::Arg("memory", "cpu_to_acl"),
         mlvc::Profiler::Arg("reason", IsQScaleInput(input_name) ? "q_scale_acl_cache_upload"
                                                                 : "stage_input_upload"),
         mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(copy.bytes)),
         mlvc::Profiler::Arg("residency", handle.ResidencyString())});
  }
  return InputView{handle.AclView(), false};
}

void StageOutputWorkspace::CopyStageTensorFromDevice(std::string_view stage_name,
                                                     std::string_view tensor_name,
                                                     mlvc::PinnedHostBuffer* buffer,
                                                     std::size_t* offset, TensorBytesView* view,
                                                     mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE(std::string("CopyStageTensorFromDevice.") + std::string(stage_name) + "." +
                     std::string(tensor_name));
  mlvc::Check(binding_mode_ == StageOutputBindingMode::kAclMirror,
              "ACL mirror is required for device tensor copies");
  Entry* entry = Find(stage_name);
  const std::size_t index = FindOutputIndex(*entry, tensor_name);
  const auto begin = std::chrono::steady_clock::now();
  CopyAclTensorToPinned(entry->handles[index], buffer, offset, view);
  const auto end = std::chrono::steady_clock::now();
  if (profiler != nullptr) {
    AddProfileEventWithArgs(
        profiler, "copy.d2h.stage_tensor." + entry->stage_name + "." + std::string(tensor_name),
        "copy", "copy", profiler->StartMs(begin), profiler->DurationMs(begin, end),
        {mlvc::Profiler::Arg("stage", entry->stage_name),
         mlvc::Profiler::Arg("tensor", std::string(tensor_name)),
         mlvc::Profiler::Arg("reason", "entropy_encode"),
         mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(view->byte_count)),
         mlvc::Profiler::Arg("memory", "acl_to_pinned_cpu"),
         mlvc::Profiler::Arg("residency", entry->handles[index].ResidencyString())});
  }
}

bool StageOutputWorkspace::BuildEncodeYIndexesToPinned(
    std::string_view spatial_stage, int part, float force_zero_thres,
    mlvc::PinnedHostBuffer* buffer, std::size_t* offset, TensorBytesView* combined_view,
    TensorBytesView* keep_view, mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE(std::string("BuildEncodeYIndexesToPinned.") + std::string(spatial_stage) +
                     ".part_" + std::to_string(part));
  if (binding_mode_ != StageOutputBindingMode::kAclMirror || !mlvc::EncodeIndexAclAvailable()) {
    return false;
  }
  mlvc::Check(part >= 0 && part < static_cast<int>(kYResidualPartNames.size()),
              "invalid entropy part index");
  mlvc::Check(
      buffer != nullptr && offset != nullptr && combined_view != nullptr && keep_view != nullptr,
      "ACL encode index output arguments must not be null");
  Entry* entry = Find(spatial_stage);
  const std::size_t symbol_index = FindOutputIndex(*entry, kYResidualPartNames[part]);
  const std::size_t scale_index = FindOutputIndex(*entry, kScalePartNames[part]);
  mlvc::TensorHandle& symbol_handle = entry->handles[symbol_index];
  mlvc::TensorHandle& scale_handle = entry->handles[scale_index];
  if (!symbol_handle.has_acl_buffer() || !symbol_handle.acl_valid() ||
      !scale_handle.has_acl_buffer() || !scale_handle.acl_valid()) {
    return false;
  }
  TensorData& symbol_tensor = entry->tensors[symbol_index];
  TensorData& scale_tensor = entry->tensors[scale_index];
  if ((symbol_tensor.dtype != mlvc::DataType::kInt8 &&
       symbol_tensor.dtype != mlvc::DataType::kFloat16) ||
      scale_tensor.dtype != mlvc::DataType::kFloat16 ||
      symbol_tensor.Elements() != scale_tensor.Elements()) {
    return false;
  }

  const std::size_t elements = symbol_tensor.Elements();
  const std::size_t combined_bytes = elements * mlvc::ElementSize(mlvc::DataType::kInt16);
  const std::size_t keep_bytes = elements * mlvc::ElementSize(mlvc::DataType::kUInt8);
  const std::size_t next_offset = *offset + combined_bytes + keep_bytes;
  mlvc::Check(next_offset <= buffer->bytes(), "async pinned entropy slot is too small");
  if (encode_index_combined_.bytes() < combined_bytes) {
    encode_index_combined_.Allocate(combined_bytes);
  }
  if (encode_index_keep_.bytes() < keep_bytes) {
    encode_index_keep_.Allocate(keep_bytes);
  }

  const auto begin = std::chrono::steady_clock::now();
  mlvc::BuildIndexEncodeAcl(
      symbol_handle.AclView().data(), symbol_tensor.dtype, scale_handle.AclView().data(),
      scale_tensor.dtype, symbol_tensor.shape.dims().data(), symbol_tensor.shape.dims().size(),
      elements, force_zero_thres, static_cast<int16_t*>(encode_index_combined_.data()),
      static_cast<uint8_t*>(encode_index_keep_.data()), g_acl_user_compute_stream, profiler);

  auto* destination = static_cast<uint8_t*>(buffer->data()) + *offset;
  CheckAclStatus(aclrtMemcpy(destination, combined_bytes, encode_index_combined_.data(),
                             combined_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy entropy Y combined index D2H");
  CheckAclStatus(aclrtMemcpy(destination + combined_bytes, keep_bytes, encode_index_keep_.data(),
                             keep_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy entropy Y keep mask D2H");
  *combined_view =
      TensorBytesView{symbol_tensor.shape, mlvc::DataType::kInt16, destination, combined_bytes};
  *keep_view = TensorBytesView{symbol_tensor.shape, mlvc::DataType::kUInt8,
                               destination + combined_bytes, keep_bytes};
  *offset = next_offset;
  const auto end = std::chrono::steady_clock::now();
  if (profiler != nullptr) {
    AddProfileEventWithArgs(
        profiler,
        "copy.d2h.entropy_y_index." + std::string(spatial_stage) + ".part_" + std::to_string(part),
        "copy", "copy", profiler->StartMs(begin), profiler->DurationMs(begin, end),
        {mlvc::Profiler::Arg("stage", std::string(spatial_stage)),
         mlvc::Profiler::Arg("part", part), mlvc::Profiler::Arg("direction", "D2H"),
         mlvc::Profiler::Arg("memory", "acl_to_pinned_cpu"),
         mlvc::Profiler::Arg("reason", "entropy_encode_prebuilt_y_index"),
         mlvc::Profiler::Arg("symbol_dtype", mlvc::DataTypeName(symbol_tensor.dtype)),
         mlvc::Profiler::Arg("scale_dtype", mlvc::DataTypeName(scale_tensor.dtype)),
         mlvc::Profiler::Arg("elements", static_cast<uint64_t>(elements)),
         mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(combined_bytes + keep_bytes))});
  }
  return true;
}

void StageOutputWorkspace::SynchronizeCopyStream(mlvc::Profiler*, std::string_view) {}

void StageOutputWorkspace::set_acl_user_compute_stream(void* stream) {
  g_acl_user_compute_stream = stream;
}

StageOutputWorkspace::Entry* StageOutputWorkspace::Find(std::string_view stage_name) {
  const Entry* entry = static_cast<const StageOutputWorkspace&>(*this).FindOrNull(stage_name);
  if (entry != nullptr) {
    return const_cast<Entry*>(entry);
  }
  throw mlvc::Error("missing stage output workspace: " + std::string(stage_name));
}

const StageOutputWorkspace::Entry* StageOutputWorkspace::Find(std::string_view stage_name) const {
  const Entry* entry = FindOrNull(stage_name);
  if (entry != nullptr) {
    return entry;
  }
  throw mlvc::Error("missing stage output workspace: " + std::string(stage_name));
}

const StageOutputWorkspace::Entry* StageOutputWorkspace::FindOrNull(
    std::string_view stage_name) const {
  for (const Entry& entry : entries_) {
    if (entry.stage_name == stage_name) {
      return &entry;
    }
  }
  return nullptr;
}

}  // namespace mlvc::codec
