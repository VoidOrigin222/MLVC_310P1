#ifndef ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_STATE_H_
#define ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_STATE_H_

#include <ulbvc/application/stream/ulbvc_stream.h>
#include <ulbvc/codec/detail/frame/reference_state.h>
#include <ulbvc/codec/detail/stage/stage_runner.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/codec/ulbvc_entropy.h>
#include <ulbvc/framework/profiler.h>

#include <ulbvc/transport/rtcp_session.h>

#include <cstdint>
#include <map>
#include <vector>

namespace ulbvc::codec {

struct EncodeFrameDecision {
  UlbvcFrameType frame_type = UlbvcFrameType::kPFrame;
  bool is_i_frame = false;
  bool mark_as_ltr = false;
  bool reset_reference = false;
  int ltr_reference_frame = -1;
};

class EncodeState {
 public:
  explicit EncodeState(const std::vector<int64_t>& feature_shape);

  EncodeFrameDecision BeginFrame(int frame_index, const EncodeStreamOptions& options);
  void SetForceRandomAccess(bool requested) { force_random_access_ = requested; }
  bool ApplyControl(const ulbvc::transport::UlbvcControlMessage& message, int frame_index,
                    bool defer_random_access = false);
  int EffectiveQIndex(int fallback) const { return fixed_q_index_ >= 0 ? fixed_q_index_ : fallback;
  }
  void ClearQIndexOverride() { fixed_q_index_ = -1; }
  void PrepareLtrRecovery(int frame_index, const EncodeStreamOptions& options,
                          const EncodeFrameDecision& decision);
  void UpdateAfterEncode(int frame_index, const EncodeFrameDecision& decision,
                         const RunOutput& output, ulbvc::Profiler* profiler);

  const ReferenceState& reference() const { return reference_; }
  ReferenceState& reference() { return reference_; }
  const TensorData& zero_feature() const { return zero_feature_; }
  int current_ltr_reference_frame() const { return current_ltr_reference_frame_; }

 private:
  ReferenceState reference_;
  TensorData zero_feature_;
  TensorData ltr_feature_;
  bool has_ltr_feature_ = false;
  std::map<int, TensorData> ltr_features_;
  int current_ltr_reference_frame_ = -1;
  bool force_random_access_ = false;
  int fixed_q_index_ = -1;
  int gop_override_ = -1;
};

}  // namespace ulbvc::codec

#endif  // ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_STATE_H_
