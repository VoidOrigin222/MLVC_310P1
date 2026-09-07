#ifndef MLVC_APPLICATION_STREAM_ENCODE_ENCODE_STATE_H_
#define MLVC_APPLICATION_STREAM_ENCODE_ENCODE_STATE_H_

#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/codec/detail/frame/reference_state.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/codec/tensor_data.h>
#include <mlvc/framework/profiler.h>

#include <map>
#include <cstdint>
#include <vector>

namespace mlvc::codec {

struct EncodeFrameDecision {
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  bool is_i_frame = false;
  bool mark_as_ltr = false;
};

class EncodeState {
 public:
  explicit EncodeState(const std::vector<int64_t>& feature_shape);

  EncodeFrameDecision BeginFrame(int frame_index, const EncodeStreamOptions& options);
  void PrepareLtrRecovery(int frame_index, const EncodeStreamOptions& options,
                          const EncodeFrameDecision& decision);
  void UpdateAfterEncode(int frame_index, const EncodeFrameDecision& decision,
                         const RunOutput& output, mlvc::Profiler* profiler);

  const ReferenceState& reference() const { return reference_; }
  const TensorData& zero_feature() const { return zero_feature_; }

 private:
  ReferenceState reference_;
  TensorData zero_feature_;
  TensorData ltr_feature_;
  bool has_ltr_feature_ = false;
  std::map<int, TensorData> ltr_features_;
};

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_ENCODE_ENCODE_STATE_H_
