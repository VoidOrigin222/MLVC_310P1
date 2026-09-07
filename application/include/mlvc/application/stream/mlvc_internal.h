#ifndef MLVC_APPLICATION_STREAM_MLVC_INTERNAL_H_
#define MLVC_APPLICATION_STREAM_MLVC_INTERNAL_H_

#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/codec/detail/frame/reference_state.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/stage/stage_types.h>
#include <mlvc/codec/tensor_data.h>
#include <mlvc/runtime/model_manifest.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include <mlvc/io/mlvc_bitstream.h>

#include <cstdint>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace mlvc::codec {

struct SourceInfo {
  int width = 1920;
  int height = 1080;
  double fps = 30.0;
};

struct DecodedEntropyFrame {
  int frame_index = -1;
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  int q_index = 0;
  TensorData z_raw;
  TensorData y_raw_0;
  TensorData y_raw_1;
};

struct PendingEncodedFrame {
  int frame_index = -1;
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  int q_index = 0;
  std::future<std::vector<uint8_t>> payload;
};

void SetInt32TensorValue(TensorData* tensor, int32_t value);
TensorData MakeInt32ScalarTensor(int32_t value);
SourceInfo ReadSourceInfo(const std::filesystem::path& frame_dir);
bool HasMlvcModels(const mlvc::ModelManifest& manifest);
SourceFrameGeometry ResolveSourceGeometry(const std::filesystem::path& input_video_path,
                                          const std::filesystem::path& input_frame_dir,
                                          const mlvc::TensorSpec& frame_spec, double* fps);
void ConfigureRuntimeState(mlvc::codec::StageOutputWorkspace* stage_output_workspace,
                           mlvc::StageRuntime& runtime, bool enable_stage_fusion);
StageInput BuildReferenceFeatureInput(const ReferenceState& state, const TensorData& zero_feature);
void UpdateReferenceFeature(const RunOutput& output, ReferenceState* state,
                            mlvc::Profiler* profiler);

class ScopedRuntimeState {
 public:
  ScopedRuntimeState();
  ~ScopedRuntimeState();

  ScopedRuntimeState(const ScopedRuntimeState&) = delete;
  ScopedRuntimeState& operator=(const ScopedRuntimeState&) = delete;

 private:
  std::unique_lock<std::mutex> lock_;
  mlvc::codec::StageOutputWorkspace* stage_output_workspace_ = nullptr;
  mlvc::CodecGraphExecutor* graph_executor_ = nullptr;
  void* acl_user_compute_stream_ = nullptr;
  bool skip_async_entropy_cpu_mirror_ = false;
  bool skip_async_encode_device_only_cpu_mirror_ = false;
  bool skip_async_decode_device_only_cpu_mirror_ = false;
  bool validate_acl_decode_prior_ = false;
  bool enable_stage_fusion_ = false;
};

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_MLVC_INTERNAL_H
