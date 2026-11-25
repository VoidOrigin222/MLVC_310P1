#ifndef ULBVC_APPLICATION_STREAM_ULBVC_INTERNAL_H_
#define ULBVC_APPLICATION_STREAM_ULBVC_INTERNAL_H_

#include <ulbvc/application/stream/ulbvc_stream.h>
#include <ulbvc/codec/detail/frame/reference_state.h>
#include <ulbvc/codec/detail/stage/stage_runner.h>
#include <ulbvc/codec/detail/stage/stage_runtime_state.h>
#include <ulbvc/codec/detail/stage/stage_types.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/io/ulbvc_bitstream.h>
#include <ulbvc/io/video_io.h>
#include <ulbvc/runtime/model_manifest.h>
#include <ulbvc/transport/detection_metadata.h>

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ulbvc::codec {

struct SourceInfo {
  int width = 1920;
  int height = 1080;
  double fps = 30.0;
};

struct DecodedEntropyFrame {
  int frame_index = -1;
  UlbvcFrameType frame_type = UlbvcFrameType::kPFrame;
  int q_index = 0;
  TensorData z_raw;
  TensorData y_raw_0;
  TensorData y_raw_1;
  ulbvc::io::UlbvcFrameMetadata metadata;
};

struct LatencyStats {
  uint64_t count = 0;
  double total_ms = 0.0;
  double max_ms = 0.0;
  void Add(double ms) { ++count; total_ms += ms; max_ms = std::max(max_ms, ms); }
  double average_ms() const { return count ? total_ms / count : 0.0; }
};

struct PendingEncodedFrame {
  int frame_index = -1;
  UlbvcFrameType frame_type = UlbvcFrameType::kPFrame;
  int q_index = 0;
  ulbvc::io::UlbvcFrameMetadata metadata;
  std::future<std::vector<uint8_t>> payload;
  std::chrono::steady_clock::time_point ready_at{};
  std::shared_future<ulbvc::transport::DetectionFrame> detection;
};

void SetInt32TensorValue(TensorData* tensor, int32_t value);
TensorData MakeInt32ScalarTensor(int32_t value);
SourceInfo ReadSourceInfo(const std::filesystem::path& frame_dir);
bool HasUlbvcModels(const ulbvc::ModelManifest& manifest);
SourceFrameGeometry ResolveSourceGeometry(const std::filesystem::path& input_video_path,
                                          const std::filesystem::path& input_frame_dir,
                                          const std::optional<ulbvc::io::CameraCaptureOptions>&
                                              camera_options,
                                          aclrtContext context,
                                          const ulbvc::TensorSpec& frame_spec, double* fps);
void ConfigureRuntimeState(ulbvc::codec::StageOutputWorkspace* stage_output_workspace,
                           ulbvc::StageRuntime& runtime, bool enable_stage_fusion);
StageInput BuildReferenceFeatureInput(const ReferenceState& state, const TensorData& zero_feature);
StageInput BuildWarpedReferenceFeatureInput(ReferenceState* state, const TensorData& zero_feature,
                                           int kx, int ky, TensorData* scratch);
void PrepareWarpResetReference(ulbvc::StageModelSet* models, ReferenceState* state,
                               const std::vector<int64_t>& frame_shape, bool is_i_frame,
                               ulbvc::Profiler* profiler);
void SaveWarpResetFrame(const RunOutput& output, ReferenceState* state, ulbvc::Profiler* profiler);
// Explicit opt-in diagnostic; callers invoke only in translation warp mode.
void TraceWarpReferenceState(int frame_index, ReferenceState* state);
void UpdateReferenceFeature(const RunOutput& output, ReferenceState* state,
                            ulbvc::Profiler* profiler);

class ScopedRuntimeState {
 public:
  ScopedRuntimeState();
  ~ScopedRuntimeState();

  ScopedRuntimeState(const ScopedRuntimeState&) = delete;
  ScopedRuntimeState& operator=(const ScopedRuntimeState&) = delete;

 private:
  std::unique_lock<std::mutex> lock_;
  ulbvc::codec::StageOutputWorkspace* stage_output_workspace_ = nullptr;
  ulbvc::CodecGraphExecutor* graph_executor_ = nullptr;
  void* acl_user_compute_stream_ = nullptr;
  bool skip_async_entropy_cpu_mirror_ = false;
  bool skip_async_encode_device_only_cpu_mirror_ = false;
  bool skip_async_decode_device_only_cpu_mirror_ = false;
  bool validate_acl_decode_prior_ = false;
  bool enable_stage_fusion_ = false;
};

}  // namespace ulbvc::codec

#endif  // ULBVC_APPLICATION_STREAM_ULBVC_INTERNAL_H
