#ifndef MLVC_CODEC_DETAIL_STAGE_WORKSPACE_H_
#define MLVC_CODEC_DETAIL_STAGE_WORKSPACE_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/buffer.h>
#include <mlvc/core/tensor.h>
#include <mlvc/core/tensor_handle.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/runtime/stage_runtime.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <mlvc/codec/detail/stage/constants.h>
#include <mlvc/codec/detail/stage/stage_types.h>

namespace mlvc::codec {

enum class StageOutputBindingMode {
  kCpu,
  kAclMirror,
};

struct InputView {
  mlvc::TensorView view;
  bool device_reuse = false;
};

struct StageOutputWorkspace {
  struct Entry {
    std::string stage_name;
    std::array<const std::string*, kMaxStageInputs> input_names;
    std::array<mlvc::TensorHandle, kMaxStageInputs> input_handles;
    std::array<const TensorData*, kMaxStageInputs> input_acl_sources;
    std::array<const std::string*, kMaxStageOutputs> names;
    std::array<TensorData, kMaxStageOutputs> tensors;
    std::array<mlvc::TensorHandle, kMaxStageOutputs> handles;
    std::size_t input_size = 0;
    std::size_t size = 0;
  };

  explicit StageOutputWorkspace(mlvc::StageModelSet* models, StageOutputBindingMode binding_mode);

  RunOutput Bind(const mlvc::ModelRecord& record);
  void BindOutputViews(const mlvc::ModelRecord& record, const RunOutput& output,
                       std::array<mlvc::NamedTensorView, kMaxStageOutputs>* output_views,
                       std::size_t* output_count);
  void MirrorOutputsToCpu(const mlvc::ModelRecord& record, const RunOutput& output,
                          mlvc::Profiler* profiler);
  bool CanCopyStageTensorFromDevice(std::string_view stage_name,
                                    std::string_view tensor_name) const;
  std::optional<InputView> InputViewForCpuMirror(const TensorData* tensor);
  std::optional<InputView> UploadInputToDevice(const mlvc::ModelRecord& record,
                                               const char* input_name, const TensorData& tensor,
                                               mlvc::Profiler* profiler);
  void CopyStageTensorFromDevice(std::string_view stage_name, std::string_view tensor_name,
                                 mlvc::PinnedHostBuffer* buffer, std::size_t* offset,
                                 TensorBytesView* view, mlvc::Profiler* profiler = nullptr);
  bool BuildEncodeYIndexesToPinned(std::string_view spatial_stage, int part, float force_zero_thres,
                                   mlvc::PinnedHostBuffer* buffer, std::size_t* offset,
                                   TensorBytesView* combined_view, TensorBytesView* keep_view,
                                   mlvc::Profiler* profiler = nullptr);
  void SynchronizeCopyStream(mlvc::Profiler* profiler = nullptr,
                             std::string_view reason = "copy_stream");

  StageOutputBindingMode binding_mode() const { return binding_mode_; }
  void set_acl_user_compute_stream(void* stream);

 private:
  Entry* Find(std::string_view stage_name);
  const Entry* Find(std::string_view stage_name) const;
  const Entry* FindOrNull(std::string_view stage_name) const;

  std::vector<Entry> entries_;
  StageOutputBindingMode binding_mode_ = StageOutputBindingMode::kCpu;
  mlvc::AclBuffer encode_index_combined_;
  mlvc::AclBuffer encode_index_keep_;
};

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_STAGE_WORKSPACE_H_
