#include <mlvc/application/stream/encode/encode_state.h>
#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/codec/tensor_utils.h>
#include <mlvc/core/status.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename Function>
void ExpectReject(Function&& function, const char* description) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  mlvc::Check(rejected, std::string("expected failure: ") + description);
}

void CacheLtrFrame(mlvc::codec::EncodeState* state, int frame_index,
                   const mlvc::codec::EncodeStreamOptions& options, float feature_value) {
  const auto decision = state->BeginFrame(frame_index, options);
  mlvc::Check(decision.mark_as_ltr, "test frame is not scheduled as an LTR capture");
  std::string feature_name = "feature";
  auto feature = mlvc::codec::MakeFp16Tensor({1}, feature_value);
  mlvc::codec::RunOutput output;
  output.tensors[0] = mlvc::codec::RunOutput::Entry{&feature_name, &feature, nullptr};
  output.size = 1;
  state->UpdateAfterEncode(frame_index, decision, output, nullptr);
}

}  // namespace

int main() {
  try {
    constexpr int kGop = 96;
    constexpr int kResetInterval = 32;
    mlvc::Check(mlvc::codec::ShouldResetReferenceFeature(0, kGop, kResetInterval),
                "the first I frame must reset the reference feature");
    mlvc::Check(mlvc::codec::ShouldResetReferenceFeature(31, kGop, kResetInterval),
                "a reset-interval boundary P frame must reset the reference feature");
    mlvc::Check(!mlvc::codec::ShouldResetReferenceFeature(32, kGop, kResetInterval),
                "a normal P frame must preserve the reference feature");
    mlvc::Check(mlvc::codec::ShouldResetReferenceFeature(63, kGop, kResetInterval),
                "each reset-interval boundary P frame must reset the reference feature");
    mlvc::Check(mlvc::codec::ShouldResetReferenceFeature(96, kGop, kResetInterval),
                "a GOP-boundary I frame must reset the reference feature");

    mlvc::codec::EncodeStreamOptions options;
    options.gop = kGop;
    options.reset_interval = kResetInterval;
    options.ltr_start_idx = 8;
    options.ltr_period = 16;
    options.forced_ltr_recovery_frame = 40;
    options.forced_ltr_reference_frame = 8;

    mlvc::codec::EncodeState state({1});
    CacheLtrFrame(&state, 8, options, 8.0f);
    CacheLtrFrame(&state, 16, options, 16.0f);
    CacheLtrFrame(&state, 32, options, 32.0f);

    const auto recovery = state.BeginFrame(40, options);
    mlvc::Check(recovery.frame_type == mlvc::codec::MlvcFrameType::kLtrRecovery,
                "forced recovery frame was not classified as LTR recovery");
    state.PrepareLtrRecovery(40, options, recovery);
    mlvc::Check(state.reference().feature.has_value(),
                "forced recovery did not populate the reference feature");
    std::vector<float> recovered_feature;
    mlvc::codec::Fp16ToFloat(state.reference().feature.value(), &recovered_feature);
    mlvc::Check(recovered_feature.size() == 1 && recovered_feature[0] == 8.0f,
                "forced recovery did not select the requested non-latest LTR frame");

    state.BeginFrame(96, options);
    options.forced_ltr_recovery_frame = 104;
    ExpectReject([&] { state.BeginFrame(104, options); },
                 "forced recovery using an LTR cache cleared at the GOP boundary");
    std::cout << "reference reset schedule test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_reference_reset_schedule failed: " << error.what() << "\n";
    return 1;
  }
}
