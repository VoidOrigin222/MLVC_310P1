#include <ulbvc/application/input/frame_input_queue.h>
#include <ulbvc/application/progress.h>
#include <ulbvc/application/stream/ulbvc_stream.h>
#include <ulbvc/codec/detail/frame/reference_state.h>
#include <ulbvc/codec/detail/profile/codec_profile.h>
#include <ulbvc/codec/detail/stage/constants.h>
#include <ulbvc/codec/detail/stage/stage_runner.h>
#include <ulbvc/codec/detail/stage/stage_runtime_state.h>
#include <ulbvc/codec/detail/stage/stage_types.h>
#include <ulbvc/codec/detail/tensor/tensor_utils.h>
#include <ulbvc/codec/execution_profile.h>
#include <ulbvc/codec/ulbvc_entropy.h>
#include <ulbvc/codec/ulbvc_rate_control.h>
#include <ulbvc/codec/tensor_utils.h>
#include <ulbvc/codec/translation_warp.h>
#include <ulbvc/entropy/entropy_codec.h>
#include <ulbvc/entropy/ulbvc_official_entropy.h>
#include <ulbvc/entropy/sidecar.h>
#include <ulbvc/framework/codec_graph_executor.h>
#include <ulbvc/framework/entropy_worker.h>
#include <ulbvc/framework/profile_range.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/io/frame_source.h>
#include <ulbvc/io/ulbvc_bitstream.h>
#include <ulbvc/io/udp_frame_transport.h>
#include <ulbvc/io/video_io.h>
#include <ulbvc/runtime/stage_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

#include "ulbvc/application/stream/ulbvc_internal.h"
#include "ulbvc/core/status.h"

namespace ulbvc::codec {
namespace {
std::mutex& RuntimeStateMutex() {
  static std::mutex mutex;
  return mutex;
}
}  // namespace
void SetInt32TensorValue(TensorData* tensor, int32_t value) {
  Check(tensor != nullptr, "int32 tensor output is required");
  Check(tensor->dtype == ulbvc::DataType::kInt32, "tensor is not int32");
  Check(tensor->shape.NumElements() == 1, "int32 tensor expects a scalar shape");
  auto* output = reinterpret_cast<int32_t*>(tensor->bytes.data());
  output[0] = value;
}

TensorData MakeInt32ScalarTensor(int32_t value) {
  TensorData tensor = MakeTensor({1}, ulbvc::DataType::kInt32);
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

bool HasUlbvcModels(const ulbvc::ModelManifest& manifest) {
  return manifest.HasModel("ULBVCEncoder") && manifest.HasModel("ULBVCDecoder");
}

SourceFrameGeometry ResolveSourceGeometry(const std::filesystem::path& input_video_path,
                                          const std::filesystem::path& input_frame_dir,
                                          const std::optional<ulbvc::io::CameraCaptureOptions>&
                                              camera_options,
                                          aclrtContext context,
                                          const ulbvc::TensorSpec& frame_spec, double* fps) {
  SourceInfo info = ReadSourceInfo(input_frame_dir);
  if (camera_options.has_value()) {
    info.width = camera_options->width;
    info.height = camera_options->height;
    info.fps = camera_options->fps;
  } else if (!input_video_path.empty()) {
    ulbvc::io::VideoFrameReader reader = camera_options.has_value()
                                            ? ulbvc::io::VideoFrameReader(input_video_path,
                                                                          *camera_options,
                                                                          context)
                                            : ulbvc::io::VideoFrameReader(input_video_path);
    const ulbvc::io::VideoInfo& video_info = reader.info();
    info.width = video_info.width;
    info.height = video_info.height;
    info.fps = video_info.fps;
  }
  if (fps != nullptr) {
    *fps = info.fps;
  }
  const bool aipp_camera = camera_options.has_value() && frame_spec.dtype == ulbvc::DataType::kUInt8;
  const int padded_height = aipp_camera
                                ? static_cast<int>(frame_spec.shape.at(1) * 2 / 3)
                                : static_cast<int>(frame_spec.shape.at(2));
  const int padded_width = aipp_camera
                               ? static_cast<int>(frame_spec.shape.at(2))
                               : static_cast<int>(frame_spec.shape.at(3));
  Check(info.width <= padded_width && info.height <= padded_height,
        "input frame size exceeds model tensor shape");
  return SourceFrameGeometry{info.width, info.height, padded_width, padded_height};
}

void ConfigureRuntimeState(ulbvc::codec::StageOutputWorkspace* stage_output_workspace,
                           ulbvc::StageRuntime& runtime, bool enable_stage_fusion) {
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

StageInput BuildWarpedReferenceFeatureInput(ReferenceState* state, const TensorData& zero_feature,
                                           int kx, int ky, TensorData* scratch) {
  Check(state != nullptr && scratch != nullptr, "translation warp requires reference and scratch");
  if (kx == 0 && ky == 0) return BuildReferenceFeatureInput(*state, zero_feature);
  if (state->feature_handle.has_value()) {
    state->feature_handle->MaterializeToCpu("translation_warp.reference", g_acl_user_compute_stream);
    ShiftTensorPreserveBoundaryInto(state->feature_handle->CpuView(), kx, ky, scratch);
  } else {
    ShiftTensorPreserveBoundaryInto(state->feature.has_value() ? *state->feature : zero_feature,
                                    kx, ky, scratch);
  }
  return TensorInput("ref_feature", *scratch);
}

void PrepareWarpResetReference(ulbvc::StageModelSet* models, ReferenceState* state,
                               const std::vector<int64_t>& frame_shape, bool is_i_frame,
                               ulbvc::Profiler* profiler) {
  Check(models != nullptr && state != nullptr, "warp reset requires models and reference state");
  if (is_i_frame) {
    state->ResetFrame();
    state->frame.emplace(MakeFp16Tensor(frame_shape, 0.5f));
  }
  Check(state->frame.has_value() || state->frame_handle.has_value(),
        "warp reset requires the previous reconstructed frame");
  const StageInput input = state->frame_handle.has_value()
      ? StageInput{"ref_frame", nullptr, &*state->frame_handle}
      : TensorInput("ref_frame", *state->frame);
  RunOutput adapted = RunStage(models, "ULBVCReferenceFromFrame", {input}, profiler);
  state->ResetFeature();
  ulbvc::TensorHandle handle;
  if (CloneRunOutputHandle(adapted, "ref_feature", &handle, "warp.reset_reference", profiler)) {
    state->feature_handle.emplace(std::move(handle));
  } else {
    state->feature.emplace(CloneTensor(adapted.At("ref_feature")));
  }
  state->ResetFrame();
}

void SaveWarpResetFrame(const RunOutput& output, ReferenceState* state, ulbvc::Profiler* profiler) {
  Check(state != nullptr, "warp reset requires reference state");
  state->ResetFrame();
  ulbvc::TensorHandle handle;
  if (CloneRunOutputHandle(output, "x_hat", &handle, "warp.reset_frame", profiler)) {
    state->frame_handle.emplace(std::move(handle));
  } else {
    state->frame.emplace(CloneTensor(output.At("x_hat")));
  }
}

void TraceWarpReferenceState(int frame_index, ReferenceState* state) {
  const char* trace_dir = std::getenv("ULBVC_WARP_TRACE_DIR");
  if (trace_dir == nullptr || *trace_dir == '\0') return;
  Check(state != nullptr && frame_index >= 0, "warp trace requires reference state and frame index");
  const std::filesystem::path directory(trace_dir);
  std::filesystem::create_directories(directory);
  const auto dump = [&](const char* suffix, std::optional<TensorData>& tensor,
                        std::optional<ulbvc::TensorHandle>& handle) {
    if (!tensor.has_value() && !handle.has_value()) return;
    if (handle.has_value()) {
      handle->MaterializeToCpu("translation_warp.trace", g_acl_user_compute_stream);
    }
    const ulbvc::TensorView view = handle.has_value() ? handle->CpuView() : tensor->View();
    Check(view.location() == ulbvc::MemoryLocation::kCpu && view.data() != nullptr,
          "warp trace requires materialized CPU storage");
    const auto path = directory / ("frame_" + std::to_string(frame_index) + suffix);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    Check(file.good(), "cannot open warp trace output: " + path.string());
    file.write(static_cast<const char*>(view.data()), static_cast<std::streamsize>(view.bytes()));
    Check(file.good(), "cannot write warp trace output: " + path.string());
  };
  dump(".feature.fp16", state->feature, state->feature_handle);
  dump(".reset_frame.fp16", state->frame, state->frame_handle);
}

void UpdateReferenceFeature(const RunOutput& output, ReferenceState* state,
                            ulbvc::Profiler* profiler) {
  Check(state != nullptr, "reference state is required");
  state->ResetFeature();
  ulbvc::TensorHandle feature_handle;
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

bool IsUlbvcManifest(const ulbvc::ModelManifest& manifest) { return HasUlbvcModels(manifest); }

}  // namespace ulbvc::codec
