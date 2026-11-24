#include <mlvc/application/stream/encode/encode_frame.h>
#include <mlvc/application/stream/live_quality_capture.h>
#include <mlvc/codec/detail/stage/constants.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/tensor/tensor_utils.h>
#include <mlvc/codec/tensor_utils.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "mlvc/core/status.h"

namespace mlvc::codec {

EncodeFrameProcessor::EncodeFrameProcessor(
    const EncodeStreamOptions& options, mlvc::StageModelSet* models,
    const mlvc::RuntimeSidecar* sidecar, mlvc::Profiler* profiler,
    mlvc::EntropyWorker* entropy_worker, EncodeState* state, MlvcRateController* rate_controller,
    mlvc::MlvcOfficialEntropyEncoder* entropy_encoder, EncodeDimensions dimensions, double fps,
    SourceFrameGeometry geometry, aclrtContext context)
    : options_(options),
      models_(models),
      sidecar_(sidecar),
      profiler_(profiler),
      entropy_worker_(entropy_worker),
      state_(state),
      rate_controller_(rate_controller),
      entropy_encoder_(entropy_encoder),
      dimensions_(dimensions),
      fps_(fps), context_(context),
      motion_layout_{geometry.width, geometry.height, geometry.width, geometry.height} {
  Check(models_ != nullptr && sidecar_ != nullptr && profiler_ != nullptr &&
            entropy_worker_ != nullptr && state_ != nullptr && rate_controller_ != nullptr &&
            entropy_encoder_ != nullptr,
        "encode frame processor dependencies are incomplete");
  if (options_.translation_warp) {
    if (!options_.motion_shifts_file.empty()) {
      std::ifstream input(options_.motion_shifts_file);
      Check(input.good(), "cannot open motion_shifts_file");
      std::string line;
      while (std::getline(input, line)) {
        if (line.empty() || line == "\r") continue;
        std::istringstream values(line);
        int index = -1, kx = 0, ky = 0;
        char comma1 = 0, comma2 = 0;
        Check(static_cast<bool>(values >> index >> comma1 >> kx >> comma2 >> ky) &&
                  comma1 == ',' && comma2 == ',' && (values >> std::ws).eof(),
              "motion_shifts_file must contain frame_index,kx,ky CSV rows without a header");
        Check(index == static_cast<int>(replay_shifts_.size()),
              "motion_shifts_file frame indexes must start at zero and be contiguous");
        Check(kx >= -128 && kx <= 127 && ky >= -128 && ky <= 127,
              "motion_shifts_file offsets must fit signed bytes");
        Check(index % options_.gop != 0 || (kx == 0 && ky == 0),
              "motion_shifts_file GOP starts must have zero translation");
        replay_shifts_.push_back(mlvc::motion::Translation{static_cast<int8_t>(kx),
                                                         static_cast<int8_t>(ky)});
      }
      Check(!replay_shifts_.empty(), "motion_shifts_file is empty");
      Check(options_.frame_num < 0 || replay_shifts_.size() ==
                                        static_cast<std::size_t>(options_.frame_num),
            "motion_shifts_file frame count must match frame_num");
    } else {
      Check(std::abs(fps_ - std::round(fps_)) < 1e-6,
            "online motion currently requires an integer frame rate");
      if (options_.motion_camera_nv12 ||
          options_.motion_backend == "dvpp")
        Check(motion_layout_.width >= 128 && motion_layout_.height >= 128,
              "motion proxy dimensions must be at least 128x128");
      if (options_.motion_backend == "dvpp") {
        mlvc::io::DvppH264EncoderConfig config;
        config.layout = motion_layout_;
        config.fps = static_cast<uint32_t>(std::llround(fps_));
        config.gop = static_cast<uint32_t>(options_.gop);
        config.bitrate = 8'000'000;
        dvpp_motion_estimator_ = std::make_unique<mlvc::motion::DvppTranslationEstimator>(
            context, config, options_.motion_skip_loop_filter);
      } else {
        mlvc::motion::TranslationEstimatorConfig config;
        config.width = motion_layout_.width;
        config.height = motion_layout_.height;
        config.fps = static_cast<int>(std::llround(fps_));
        config.gop = options_.gop;
        config.preset = options_.motion_x264_preset;
        config.threads = options_.motion_x264_threads;
        config.skip_loop_filter = options_.motion_skip_loop_filter;
        motion_estimator_ = std::make_unique<mlvc::motion::TranslationEstimator>(config);
      }
    }
  }
}

EncodeFrameProcessor::~EncodeFrameProcessor() {
  // Failed consumers can leave entropy futures unretired. Their tasks capture
  // this processor and must finish before its state/dependencies are destroyed.
  std::unique_lock<std::mutex> lock(entropy_tasks_.mutex);
  entropy_tasks_.condition.wait(lock, [this] { return entropy_tasks_.active == 0; });
}

mlvc::motion::Translation EncodeFrameProcessor::EstimateMotion(
    const mlvc::app::InputFrame& frame, bool random_access) {
  Check(frame.frame_index == next_motion_frame_, "motion frames must be sequential");
  Check(!options_.motion_camera_nv12 || frame.motion_nv12,
        "configured motion proxy requires its prepared NV12 input");
  const auto begin = std::chrono::steady_clock::now();
  if (frame.ready_at != std::chrono::steady_clock::time_point{})
    ready_to_motion_.Add(std::chrono::duration<double, std::milli>(begin - frame.ready_at).count());
  mlvc::motion::Translation result;
  if (dvpp_motion_estimator_) {
    if (frame.motion_nv12)
      result = dvpp_motion_estimator_->EstimateNv12Host(frame.motion_nv12->data(),
          frame.motion_nv12->size(), frame.frame_index, random_access);
    else
      result = dvpp_motion_estimator_->Estimate(*frame.frame, frame.frame_index, random_access);
  }
  else if (motion_estimator_) {
    if (frame.motion_nv12)
      result = motion_estimator_->EstimateNv12(frame.motion_nv12->data(),
          frame.motion_nv12->size(), motion_layout_, frame.frame_index, random_access);
    else
      result = motion_estimator_->Estimate(*frame.frame, frame.frame_index, random_access);
  }
  else {
    Check(static_cast<std::size_t>(frame.frame_index) < replay_shifts_.size(),
          "motion_shifts_file ended before the input video");
    result = replay_shifts_[frame.frame_index];
  }
  motion_work_.Add(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
  ++next_motion_frame_;
  return result;
}

void EncodeFrameProcessor::PrepareMotion(const std::shared_ptr<mlvc::DataObject>& data) {
  auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
  Check(packet && packet->frame().frame, "motion worker received an empty frame");
  if (next_motion_frame_ == 0)
    Check(aclrtSetCurrentContext(context_) == ACL_SUCCESS, "motion worker cannot set ACL context");
  packet->motion = EstimateMotion(packet->frame(), packet->frame().frame_index % options_.gop == 0);
  packet->motion_ready_at = std::chrono::steady_clock::now();
}

void EncodeFrameProcessor::ValidateMotionFrameCount(int frames) const {
  if (options_.translation_warp)
    Check(next_motion_frame_ == frames, "motion frame count differs from encoded frame count");
  if (!options_.motion_shifts_file.empty())
    Check(replay_shifts_.size() == static_cast<std::size_t>(frames),
          "motion_shifts_file has unused rows because the input ended early");
}

PendingEncodedFrame EncodeFrameProcessor::Process(const std::shared_ptr<mlvc::DataObject>& data,
                                                  int expected_frame_index) {
  auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
  Check(packet != nullptr, "frame pipeline returned an unexpected data object");
  mlvc::app::InputFrame& prepared_frame = packet->frame();
  Check(prepared_frame.frame != nullptr, "async input returned an empty frame");
  const int frame_index = prepared_frame.frame_index;
  Check(frame_index == expected_frame_index, "async input frame order mismatch");

  static mlvc::app::LiveQualityCapture quality_capture;
  quality_capture.Reference(frame_index, *prepared_frame.frame, prepared_frame.motion_nv12);

  const EncodeFrameDecision decision = state_->BeginFrame(frame_index, options_);
  if (options_.translation_warp && decision.reset_reference) {
    PrepareWarpResetReference(models_, &state_->reference(), prepared_frame.frame->shape.dims(),
                              decision.is_i_frame, profiler_);
  }
  mlvc::motion::Translation translation;
  if (options_.translation_warp) {
    if (options_.motion_prefetch_frames > 0) {
      Check(packet->motion.has_value(), "motion prefetch returned a frame without motion");
      translation = *packet->motion;
    } else {
      translation = EstimateMotion(prepared_frame, decision.is_i_frame);
    }
    if (decision.is_i_frame) translation.kx = translation.ky = 0;
    if (translation.kx != 0 || translation.ky != 0) ++motion_nonzero_frames_;
  }
  state_->PrepareLtrRecovery(frame_index, options_, decision);
  const int frame_adaptation_index = kIndexMap[(frame_index + 1) % 8];
  const int base_q_index =
      rate_controller_->SolveQIndex(static_cast<double>(frame_index) / fps_, decision.frame_type);
  const int frame_q_index = std::clamp(
      base_q_index +
          (decision.frame_type == MlvcFrameType::kLtrRecovery ? options_.ltr_qp_shift : 0),
      options_.min_qp, options_.max_qp);
  const int controlled_q_index =
      std::clamp(state_->EffectiveQIndex(frame_q_index), options_.min_qp, options_.max_qp);
  const int q_index_shifted = sidecar_->ShiftedQp(controlled_q_index, frame_adaptation_index);
  TensorData q_index_shifted_tensor = MakeInt32ScalarTensor(q_index_shifted);
  const StageInput ref_feature_input = options_.translation_warp
      ? BuildWarpedReferenceFeatureInput(&state_->reference(), state_->zero_feature(),
                                          translation.kx, translation.ky, &warped_reference_)
      : BuildReferenceFeatureInput(state_->reference(), state_->zero_feature());
  RunOutput encoder_output = RunStage(models_, "MLVCEncoder",
                                      {TensorInput("x", *prepared_frame.frame), ref_feature_input,
                                       TensorInput("q_index_shifted", q_index_shifted_tensor)},
                                      profiler_);
  // Only resets outside the GOP need a reconstructed reference. The encoder
  // normally exports no x_hat, so reconstruct precisely the preceding frame
  // using the same latents, warped reference, and model QP as the decoder.
  const int next_frame = frame_index + 1;
  if (options_.translation_warp && next_frame % options_.reset_interval == 0 &&
      next_frame % options_.gop != 0) {
    RunOutput reconstruction = RunStage(models_, "MLVCDecoder",
        {TensorInput("z_raw", encoder_output.At("z_raw")),
         TensorInput("y_raw_0", encoder_output.At("y_raw_0")),
         TensorInput("y_raw_1", encoder_output.At("y_raw_1")), ref_feature_input,
         TensorInput("q_index_shifted", q_index_shifted_tensor)}, profiler_);
    SaveWarpResetFrame(reconstruction, &state_->reference(), profiler_);
  }

  const TensorData encoder_z_raw = CloneTensor(encoder_output.At("z_raw"));
  const TensorData encoder_y_raw_0 = CloneTensor(encoder_output.At("y_raw_0"));
  const TensorData encoder_y_raw_1 = CloneTensor(encoder_output.At("y_raw_1"));
  {
    std::lock_guard<std::mutex> lock(entropy_tasks_.mutex);
    ++entropy_tasks_.active;
  }
  std::future<std::vector<uint8_t>> entropy_future;
  try {
    entropy_future = entropy_worker_->Submit(
      [this, encoder_z_raw, encoder_y_raw_0, encoder_y_raw_1, controlled_q_index]() mutable {
        struct Completion {
          EntropyTasks* tasks;
          ~Completion() { tasks->Complete(); }
        } completion{&entropy_tasks_};
        mlvc::ScopedCpuTimer timer(profiler_, "entropy.rans_encode");
        std::vector<int8_t> z_symbols;
        std::vector<int8_t> y_symbols_0;
        std::vector<int8_t> y_symbols_1;
        std::vector<uint8_t> scales_0;
        std::vector<uint8_t> scales_1;
        TensorToInt8Symbols(encoder_z_raw, &z_symbols);
        TensorToInt8Symbols(encoder_y_raw_0, &y_symbols_0);
        TensorToInt8Symbols(encoder_y_raw_1, &y_symbols_1);
        BuildMlvcScaleIndexesFromZRaw(encoder_z_raw, dimensions_.y_channels, dimensions_.y_height,
                                      dimensions_.y_width, kMlvcChannelRepeat, &scales_0,
                                      &scales_1);
        std::vector<uint8_t> result;
        entropy_encoder_->Encode(
            z_symbols, y_symbols_0, y_symbols_1, scales_0, scales_1, controlled_q_index,
            static_cast<int>(z_symbols.size() / (static_cast<std::size_t>(dimensions_.z_height) *
                                                 dimensions_.z_width)),
            dimensions_.z_height, dimensions_.z_width, &result);
        return result;
      });
  } catch (...) {
    entropy_tasks_.Complete();
    throw;
  }

  state_->UpdateAfterEncode(frame_index, decision, encoder_output, profiler_);
  if (options_.translation_warp) TraceWarpReferenceState(frame_index, &state_->reference());
  const MlvcFrameType frame_type = decision.frame_type;
  mlvc::io::MlvcFrameMetadata metadata;
  metadata.explicit_metadata = true;
  metadata.translation_warp = options_.translation_warp;
  metadata.kx = translation.kx;
  metadata.ky = translation.ky;
  metadata.model_q_index = q_index_shifted;
  metadata.pts = static_cast<int64_t>(std::llround(
      static_cast<long double>(frame_index) * 90000.0L / static_cast<long double>(fps_)));
  metadata.unit_flags = mlvc::transport::kEfuCrcPresent;
  if (decision.is_i_frame) {
    metadata.unit_flags |= mlvc::transport::kEfuRandomAccess |
                           mlvc::transport::kEfuResetReference;
    metadata.short_ref_frame_id = mlvc::transport::kMlvcNoReference;
    metadata.long_ref_frame_id = mlvc::transport::kMlvcNoReference;
  } else if (frame_type == MlvcFrameType::kLtrRecovery) {
    metadata.long_ref_frame_id = static_cast<uint32_t>(state_->current_ltr_reference_frame());
    metadata.short_ref_frame_id = mlvc::transport::kMlvcNoReference;
  } else {
    metadata.short_ref_frame_id = decision.reset_reference && !options_.translation_warp
                                      ? mlvc::transport::kMlvcNoReference
                                      : static_cast<uint32_t>(frame_index - 1);
  }
  if (decision.reset_reference && (!options_.translation_warp || decision.is_i_frame))
    metadata.unit_flags |= mlvc::transport::kEfuResetReference;
  if (decision.mark_as_ltr) metadata.unit_flags |= mlvc::transport::kEfuStoreAsLtr;
  const auto ready_at = prepared_frame.ready_at;
  if (ready_at != std::chrono::steady_clock::time_point{})
    ready_to_encode_.Add(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - ready_at).count());
  packet->Release();
  return PendingEncodedFrame{frame_index, frame_type, controlled_q_index, metadata,
                             std::move(entropy_future), ready_at};
}

}  // namespace mlvc::codec
