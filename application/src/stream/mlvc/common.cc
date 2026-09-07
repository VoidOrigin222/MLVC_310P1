#include <mlvc/application/input/frame_input_queue.h>
#include <mlvc/application/progress.h>
#include <mlvc/codec/execution_profile.h>
#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/codec/mlvc_rate_control.h>
#include <mlvc/codec/tensor_utils.h>
#include <mlvc/entropy/entropy_codec.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/framework/codec_graph_executor.h>
#include <mlvc/framework/entropy_worker.h>
#include <mlvc/framework/profile_range.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/io/frame_source.h>
#include <mlvc/io/mlvc_bitstream.h>
#include <mlvc/io/udp_frame_transport.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/stage_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mlvc/codec/detail/profile/codec_profile.h>
#include <mlvc/codec/detail/stage/constants.h>
#include <mlvc/codec/detail/frame/reference_state.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include <mlvc/codec/detail/stage/stage_types.h>
#include <mlvc/codec/detail/tensor/tensor_utils.h>
#include "mlvc/core/status.h"

#include "mlvc/application/stream/mlvc_internal.h"

namespace mlvc::codec {
namespace {
std::mutex& RuntimeStateMutex() {
  static std::mutex mutex;
  return mutex;
}
}  // namespace
void SetInt32TensorValue(TensorData* tensor, int32_t value) {
  Check(tensor != nullptr, "int32 tensor output is required");
  Check(tensor->dtype == mlvc::DataType::kInt32, "tensor is not int32");
  Check(tensor->shape.NumElements() == 1, "int32 tensor expects a scalar shape");
  auto* output = reinterpret_cast<int32_t*>(tensor->bytes.data());
  output[0] = value;
}

TensorData MakeInt32ScalarTensor(int32_t value) {
  TensorData tensor = MakeTensor({1}, mlvc::DataType::kInt32);
  SetInt32TensorValue(&tensor, value);
  return tensor;
}

SourceInfo ReadSourceInfo(const std::filesystem::path& frame_dir) {
  SourceInfo info;
  if (frame_dir.empty()) {
    return info;
  }
  const std::filesystem::path source_info_path = frame_dir / "source_info.txt";
  std::ifstream input(source_info_path);
  if (!input.good()) {
    return info;
  }
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, pos);
    const std::string value = line.substr(pos + 1);
    if (key == "width") {
      info.width = std::stoi(value);
    } else if (key == "height") {
      info.height = std::stoi(value);
    } else if (key == "fps") {
      info.fps = std::stod(value);
    }
  }
  return info;
}

bool HasMlvcModels(const mlvc::ModelManifest& manifest) {
  return manifest.HasModel("MLVCEncoder") && manifest.HasModel("MLVCDecoder");
}

SourceFrameGeometry ResolveSourceGeometry(const std::filesystem::path& input_video_path,
                                          const std::filesystem::path& input_frame_dir,
                                          const mlvc::TensorSpec& frame_spec, double* fps) {
  SourceInfo info = ReadSourceInfo(input_frame_dir);
  if (!input_video_path.empty()) {
    mlvc::io::VideoFrameReader reader(input_video_path);
    const mlvc::io::VideoInfo& video_info = reader.info();
    info.width = video_info.width;
    info.height = video_info.height;
    info.fps = video_info.fps;
  }
  if (fps != nullptr) {
    *fps = info.fps;
  }
  const int padded_height = static_cast<int>(frame_spec.shape.at(2));
  const int padded_width = static_cast<int>(frame_spec.shape.at(3));
  Check(info.width <= padded_width && info.height <= padded_height,
        "input frame size exceeds model tensor shape");
  return SourceFrameGeometry{info.width, info.height, padded_width, padded_height};
}

void ConfigureRuntimeState(mlvc::codec::StageOutputWorkspace* stage_output_workspace,
                           mlvc::StageRuntime& runtime, bool enable_stage_fusion) {
  stage_output_workspace->set_acl_user_compute_stream(runtime.stream());
  g_stage_output_workspace = stage_output_workspace;
  g_skip_async_entropy_cpu_mirror = false;
  g_skip_async_encode_device_only_cpu_mirror = false;
  g_skip_async_decode_device_only_cpu_mirror = false;
  g_acl_user_compute_stream = runtime.stream();
  g_validate_acl_decode_prior = false;
  g_enable_stage_fusion = enable_stage_fusion;
}

StageInput BuildReferenceFeatureInput(const ReferenceState& state, const TensorData& zero_feature) {
  if (state.feature_handle.has_value()) {
    return StageInput{"ref_feature", nullptr, &*state.feature_handle};
  }
  if (state.feature.has_value()) {
    return TensorInput("ref_feature", *state.feature);
  }
  return TensorInput("ref_feature", zero_feature);
}

void UpdateReferenceFeature(const RunOutput& output, ReferenceState* state,
                            mlvc::Profiler* profiler) {
  Check(state != nullptr, "reference state is required");
  state->ResetFeature();
  mlvc::TensorHandle feature_handle;
  if (CloneRunOutputHandle(output, "feature", &feature_handle, "reference_feature", profiler)) {
    state->feature_handle.emplace(std::move(feature_handle));
    return;
  }
  state->feature.emplace(output.At("feature"));
}

ScopedRuntimeState::ScopedRuntimeState()
    : lock_(RuntimeStateMutex()),
      stage_output_workspace_(g_stage_output_workspace),
      graph_executor_(g_codec_graph_executor),
      acl_user_compute_stream_(g_acl_user_compute_stream),
      skip_async_entropy_cpu_mirror_(g_skip_async_entropy_cpu_mirror),
      skip_async_encode_device_only_cpu_mirror_(g_skip_async_encode_device_only_cpu_mirror),
      skip_async_decode_device_only_cpu_mirror_(g_skip_async_decode_device_only_cpu_mirror),
      validate_acl_decode_prior_(g_validate_acl_decode_prior),
      enable_stage_fusion_(g_enable_stage_fusion) {}

ScopedRuntimeState::~ScopedRuntimeState() {
  g_stage_output_workspace = stage_output_workspace_;
  g_codec_graph_executor = graph_executor_;
  g_acl_user_compute_stream = acl_user_compute_stream_;
  g_skip_async_entropy_cpu_mirror = skip_async_entropy_cpu_mirror_;
  g_skip_async_encode_device_only_cpu_mirror = skip_async_encode_device_only_cpu_mirror_;
  g_skip_async_decode_device_only_cpu_mirror = skip_async_decode_device_only_cpu_mirror_;
  g_validate_acl_decode_prior = validate_acl_decode_prior_;
  g_enable_stage_fusion = enable_stage_fusion_;
}


bool IsMlvcManifest(const mlvc::ModelManifest& manifest) {
  return HasMlvcModels(manifest);
}

}  // namespace mlvc::codec
