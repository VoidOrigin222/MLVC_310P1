#include <mlvc/application/stream/encode/encode_state.h>

#include <mlvc/codec/tensor_utils.h>

#include "mlvc/core/status.h"
#include "mlvc/application/stream/mlvc_internal.h"

namespace mlvc::codec {

EncodeState::EncodeState(const std::vector<int64_t>& feature_shape)
    : zero_feature_(MakeFp16Tensor(feature_shape, 0.0f)),
      ltr_feature_(CloneTensor(zero_feature_)) {}

EncodeFrameDecision EncodeState::BeginFrame(int frame_index,
                                             const EncodeStreamOptions& options) {
  const bool is_i_frame = IsMlvcIFrame(frame_index, options.gop, options.reset_interval);
  const int gop_cycle_index = options.gop > 0 ? frame_index % options.gop : frame_index;
  if (ShouldResetReferenceFeature(frame_index, options.gop, options.reset_interval)) {
    reference_.ResetFeature();
  }
  if (is_i_frame) {
    has_ltr_feature_ = false;
    FillFp16Tensor(0.0f, &ltr_feature_);
  }
  const bool mark_as_ltr =
      options.ltr_period > 0 &&
      (gop_cycle_index == options.ltr_start_idx ||
       (gop_cycle_index > options.ltr_start_idx && gop_cycle_index % options.ltr_period == 0));
  const bool forced_ltr_recovery = frame_index == options.forced_ltr_recovery_frame;
  const bool use_ltr_recovery = !is_i_frame && has_ltr_feature_ &&
                                (forced_ltr_recovery || mark_as_ltr);
  return EncodeFrameDecision{
      is_i_frame ? MlvcFrameType::kIFrame
                 : (use_ltr_recovery ? MlvcFrameType::kLtrRecovery : MlvcFrameType::kPFrame),
      is_i_frame, mark_as_ltr};
}

void EncodeState::PrepareLtrRecovery(int frame_index, const EncodeStreamOptions& options,
                                     const EncodeFrameDecision& decision) {
  if (decision.frame_type != MlvcFrameType::kLtrRecovery) return;
  const bool forced = frame_index == options.forced_ltr_recovery_frame;
  const int reference_frame = forced && options.forced_ltr_reference_frame >= 0
                                  ? options.forced_ltr_reference_frame
                                  : ltr_features_.rbegin()->first;
  const auto ltr_it = ltr_features_.find(reference_frame);
  Check(ltr_it != ltr_features_.end(),
        "requested LTR reference frame is not cached: " + std::to_string(reference_frame));
  reference_.feature = CloneTensor(ltr_it->second);
  reference_.feature_handle.reset();
}

void EncodeState::UpdateAfterEncode(int frame_index, const EncodeFrameDecision& decision,
                                    const RunOutput& output, mlvc::Profiler* profiler) {
  UpdateReferenceFeature(output, &reference_, profiler);
  if (decision.mark_as_ltr) {
    CloneTensorInto(output.At("feature"), &ltr_feature_);
    ltr_features_[frame_index] = CloneTensor(ltr_feature_);
    has_ltr_feature_ = true;
  }
}

}  // namespace mlvc::codec
