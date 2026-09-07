#include <mlvc/application/stream/encode/encode_frame.h>

#include <algorithm>

#include <mlvc/codec/detail/stage/constants.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/tensor/tensor_utils.h>
#include <mlvc/codec/tensor_utils.h>

#include "mlvc/core/status.h"

namespace mlvc::codec {

EncodeFrameProcessor::EncodeFrameProcessor(
    const EncodeStreamOptions& options, mlvc::StageModelSet* models,
    const mlvc::RuntimeSidecar* sidecar, mlvc::Profiler* profiler,
    mlvc::EntropyWorker* entropy_worker, EncodeState* state,
    MlvcRateController* rate_controller,
    mlvc::MlvcOfficialEntropyEncoder* entropy_encoder,
    EncodeDimensions dimensions, double fps)
    : options_(options),
      models_(models),
      sidecar_(sidecar),
      profiler_(profiler),
      entropy_worker_(entropy_worker),
      state_(state),
      rate_controller_(rate_controller),
      entropy_encoder_(entropy_encoder),
      dimensions_(dimensions),
      fps_(fps) {
  Check(models_ != nullptr && sidecar_ != nullptr && profiler_ != nullptr &&
            entropy_worker_ != nullptr && state_ != nullptr && rate_controller_ != nullptr &&
            entropy_encoder_ != nullptr,
        "encode frame processor dependencies are incomplete");
}

PendingEncodedFrame EncodeFrameProcessor::Process(
    const std::shared_ptr<mlvc::DataObject>& data, int expected_frame_index) {
  auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
  Check(packet != nullptr, "frame pipeline returned an unexpected data object");
  mlvc::app::InputFrame& prepared_frame = packet->frame();
  Check(prepared_frame.frame != nullptr, "async input returned an empty frame");
  const int frame_index = prepared_frame.frame_index;
  Check(frame_index == expected_frame_index, "async input frame order mismatch");

  const EncodeFrameDecision decision = state_->BeginFrame(frame_index, options_);
  state_->PrepareLtrRecovery(frame_index, options_, decision);
  const int frame_adaptation_index = kIndexMap[(frame_index + 1) % 8];
  const int base_q_index =
      rate_controller_->SolveQIndex(static_cast<double>(frame_index) / fps_, decision.frame_type);
  const int frame_q_index = std::min(
      63, base_q_index +
              (decision.frame_type == MlvcFrameType::kLtrRecovery ? options_.ltr_qp_shift : 0));
  const int q_index_shifted = sidecar_->ShiftedQp(frame_q_index, frame_adaptation_index);
  TensorData q_index_shifted_tensor = MakeInt32ScalarTensor(q_index_shifted);
  const StageInput ref_feature_input =
      BuildReferenceFeatureInput(state_->reference(), state_->zero_feature());
  RunOutput encoder_output =
      RunStage(models_, "MLVCEncoder",
               {TensorInput("x", *prepared_frame.frame), ref_feature_input,
                TensorInput("q_index_shifted", q_index_shifted_tensor)},
               profiler_);

  const TensorData encoder_z_raw = CloneTensor(encoder_output.At("z_raw"));
  const TensorData encoder_y_raw_0 = CloneTensor(encoder_output.At("y_raw_0"));
  const TensorData encoder_y_raw_1 = CloneTensor(encoder_output.At("y_raw_1"));
  std::future<std::vector<uint8_t>> entropy_future = entropy_worker_->Submit(
      [this, encoder_z_raw, encoder_y_raw_0, encoder_y_raw_1, frame_q_index]() mutable {
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
            z_symbols, y_symbols_0, y_symbols_1, scales_0, scales_1, frame_q_index,
            static_cast<int>(z_symbols.size() /
                             (static_cast<std::size_t>(dimensions_.z_height) * dimensions_.z_width)),
            dimensions_.z_height, dimensions_.z_width, &result);
        return result;
      });

  state_->UpdateAfterEncode(frame_index, decision, encoder_output, profiler_);
  const MlvcFrameType frame_type = decision.frame_type;
  packet->Release();
  return PendingEncodedFrame{frame_index, frame_type, frame_q_index, std::move(entropy_future)};
}

}  // namespace mlvc::codec
