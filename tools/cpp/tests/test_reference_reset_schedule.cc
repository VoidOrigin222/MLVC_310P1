#include <ulbvc/application/stream/encode/encode_state.h>
#include <ulbvc/codec/ulbvc_entropy.h>
#include <ulbvc/codec/tensor_utils.h>
#include <ulbvc/core/status.h>

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
  ulbvc::Check(rejected, std::string("expected failure: ") + description);
}

void CacheLtrFrame(ulbvc::codec::EncodeState* state, int frame_index,
                   const ulbvc::codec::EncodeStreamOptions& options, float feature_value) {
  const auto decision = state->BeginFrame(frame_index, options);
  ulbvc::Check(decision.mark_as_ltr, "test frame is not scheduled as an LTR capture");
  std::string feature_name = "feature";
  auto feature = ulbvc::codec::MakeFp16Tensor({1}, feature_value);
  ulbvc::codec::RunOutput output;
  output.tensors[0] = ulbvc::codec::RunOutput::Entry{&feature_name, &feature, nullptr};
  output.size = 1;
  state->UpdateAfterEncode(frame_index, decision, output, nullptr);
}

}  // namespace

int main() {
  try {
    constexpr int kGop = 96;
    constexpr int kResetInterval = 32;
    ulbvc::Check(ulbvc::codec::ShouldResetReferenceFeature(0, kGop, kResetInterval),
                "the first I frame must reset the reference feature");
    ulbvc::Check(ulbvc::codec::ShouldResetReferenceFeature(31, kGop, kResetInterval),
                "a reset-interval boundary P frame must reset the reference feature");
    ulbvc::Check(!ulbvc::codec::ShouldResetReferenceFeature(32, kGop, kResetInterval),
                "a normal P frame must preserve the reference feature");
    ulbvc::Check(ulbvc::codec::ShouldResetReferenceFeature(63, kGop, kResetInterval),
                "each reset-interval boundary P frame must reset the reference feature");
    ulbvc::Check(ulbvc::codec::ShouldResetReferenceFeature(96, kGop, kResetInterval),
                "a GOP-boundary I frame must reset the reference feature");

    ulbvc::codec::EncodeStreamOptions options;
    options.gop = kGop;
    options.reset_interval = kResetInterval;
    options.ltr_start_idx = 8;
    options.ltr_period = 16;
    options.forced_ltr_recovery_frame = 40;
    options.forced_ltr_reference_frame = 8;

    ulbvc::codec::EncodeState state({1});
    CacheLtrFrame(&state, 8, options, 8.0f);
    CacheLtrFrame(&state, 16, options, 16.0f);
    CacheLtrFrame(&state, 32, options, 32.0f);

    const auto recovery = state.BeginFrame(40, options);
    ulbvc::Check(recovery.frame_type == ulbvc::codec::UlbvcFrameType::kLtrRecovery,
                "forced recovery frame was not classified as LTR recovery");
    state.PrepareLtrRecovery(40, options, recovery);
    ulbvc::Check(state.reference().feature.has_value(),
                "forced recovery did not populate the reference feature");
    std::vector<float> recovered_feature;
    ulbvc::codec::Fp16ToFloat(state.reference().feature.value(), &recovered_feature);
    ulbvc::Check(recovered_feature.size() == 1 && recovered_feature[0] == 8.0f,
                "forced recovery did not select the requested non-latest LTR frame");

    state.BeginFrame(96, options);
    options.forced_ltr_recovery_frame = 104;
    ExpectReject([&] { state.BeginFrame(104, options); },
                 "forced recovery using an LTR cache cleared at the GOP boundary");
    ulbvc::codec::EncodeStreamOptions warp_options;
    warp_options.gop = kGop;
    warp_options.reset_interval = kResetInterval;
    warp_options.ltr_period = 0;
    warp_options.translation_warp = true;
    ulbvc::codec::EncodeState warp_state({1});
    ulbvc::codec::EncodeState legacy_state({1});
    auto legacy_options = warp_options;
    legacy_options.translation_warp = false;
    for (int frame = 0; frame < 193; ++frame) {
      const auto warp = warp_state.BeginFrame(frame, warp_options);
      const auto legacy = legacy_state.BeginFrame(frame, legacy_options);
      ulbvc::Check(warp.is_i_frame == (frame % kGop == 0), "warp GOP schedule changed");
      ulbvc::Check(warp.reset_reference == (frame % kResetInterval == 0),
                  "warp must reset from reconstructed frames at 0,32,64,...");
      ulbvc::Check(legacy.reset_reference ==
                      ulbvc::codec::ShouldResetReferenceFeature(frame, kGop, kResetInterval),
                  "disabling warp changed the legacy reference schedule");
    }
    std::cout << "reference reset schedule test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_reference_reset_schedule failed: " << error.what() << "\n";
    return 1;
  }
}
