#ifndef MLVC_APPLICATION_STREAM_ENCODE_ENCODE_STATE_H_
#define MLVC_APPLICATION_STREAM_ENCODE_ENCODE_STATE_H_

#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/codec/detail/frame/reference_state.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/codec/tensor_data.h>
#include <mlvc/framework/profiler.h>

#include <cstdint>
#include <map>
#include <vector>
#include <mlvc/transport/rtcp_session.h>

namespace mlvc::codec {

struct EncodeFrameDecision {
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
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
  bool ApplyControl(const mlvc::transport::MlvcControlMessage& message, int frame_index);
  int EffectiveQIndex(int fallback) const { return fixed_q_index_ >= 0 ? fixed_q_index_ : fallback; }
  void PrepareLtrRecovery(int frame_index, const EncodeStreamOptions& options,
                          const EncodeFrameDecision& decision);
  void UpdateAfterEncode(int frame_index, const EncodeFrameDecision& decision,
                         const RunOutput& output, mlvc::Profiler* profiler);

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

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_ENCODE_ENCODE_STATE_H_
