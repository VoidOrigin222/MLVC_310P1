#ifndef ULBVC_CODEC_DETAIL_STAGE_WORKSPACE_H_
#define ULBVC_CODEC_DETAIL_STAGE_WORKSPACE_H_

#include <ulbvc/codec/detail/stage/constants.h>
#include <ulbvc/codec/detail/stage/stage_types.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/buffer.h>
#include <ulbvc/core/tensor.h>
#include <ulbvc/core/tensor_handle.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/runtime/stage_runtime.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ulbvc::codec {

enum class StageOutputBindingMode {
  kCpu,
  kAclDeviceWithCpuMirror,
};

struct InputView {
  ulbvc::TensorView view;
  bool device_reuse = false;
};

struct StageOutputWorkspace {
  struct Entry {
    std::string stage_name;
    std::array<const std::string*, kMaxStageInputs> input_names;
    std::array<ulbvc::TensorHandle, kMaxStageInputs> input_handles;
    std::array<const TensorData*, kMaxStageInputs> input_acl_sources;
    std::array<const std::string*, kMaxStageOutputs> names;
    std::array<TensorData, kMaxStageOutputs> tensors;
    std::array<ulbvc::TensorHandle, kMaxStageOutputs> handles;
    std::size_t input_size = 0;
    std::size_t size = 0;
  };

  explicit StageOutputWorkspace(ulbvc::StageModelSet* models, StageOutputBindingMode binding_mode);

  RunOutput Bind(const ulbvc::ModelRecord& record);
  void BindOutputViews(const ulbvc::ModelRecord& record, const RunOutput& output,
                       std::array<ulbvc::NamedTensorView, kMaxStageOutputs>* output_views,
                       std::size_t* output_count);
  void MirrorOutputsToCpu(const ulbvc::ModelRecord& record, const RunOutput& output,
                          ulbvc::Profiler* profiler);
  bool CanCopyStageTensorFromDevice(std::string_view stage_name,
                                    std::string_view tensor_name) const;
  std::optional<InputView> InputViewForCpuMirror(const TensorData* tensor);
  std::optional<InputView> UploadInputToDevice(const ulbvc::ModelRecord& record,
                                               const char* input_name, const TensorData& tensor,
                                               ulbvc::Profiler* profiler);
  void CopyStageTensorFromDevice(std::string_view stage_name, std::string_view tensor_name,
                                 ulbvc::PinnedHostBuffer* buffer, std::size_t* offset,
                                 TensorBytesView* view, ulbvc::Profiler* profiler = nullptr);
  bool BuildEncodeYIndexesToPinned(std::string_view spatial_stage, int part, float force_zero_thres,
                                   ulbvc::PinnedHostBuffer* buffer, std::size_t* offset,
                                   TensorBytesView* combined_view, TensorBytesView* keep_view,
                                   ulbvc::Profiler* profiler = nullptr);
  void SynchronizeCopyStream(ulbvc::Profiler* profiler = nullptr,
                             std::string_view reason = "copy_stream");

  StageOutputBindingMode binding_mode() const { return binding_mode_; }
  void set_acl_user_compute_stream(void* stream);

 private:
  Entry* Find(std::string_view stage_name);
  const Entry* Find(std::string_view stage_name) const;
  const Entry* FindOrNull(std::string_view stage_name) const;

  std::vector<Entry> entries_;
  StageOutputBindingMode binding_mode_ = StageOutputBindingMode::kCpu;
  ulbvc::AclBuffer encode_index_combined_;
  ulbvc::AclBuffer encode_index_keep_;
};

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_STAGE_WORKSPACE_H_
