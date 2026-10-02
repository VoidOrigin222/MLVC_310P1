#ifndef MLVC_APPLICATION_STREAM_ENCODE_ENCODE_FRAME_H_
#define MLVC_APPLICATION_STREAM_ENCODE_ENCODE_FRAME_H_

#include <mlvc/application/pipeline/codec_frame_pipeline.h>
#include <mlvc/application/stream/encode/encode_state.h>
#include <mlvc/application/stream/mlvc_internal.h>
#include <mlvc/codec/mlvc_rate_control.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/framework/codec_graph_executor.h>
#include <mlvc/framework/entropy_worker.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/runtime/stage_runtime.h>
#include <mlvc/motion/translation_estimator.h>
#include <mlvc/motion/dvpp_translation_estimator.h>

#include <cstddef>
#include <future>
#include <memory>

namespace mlvc::codec {

struct EncodeDimensions {
  int y_channels = 0;
  int y_height = 0;
  int y_width = 0;
  int z_height = 0;
  int z_width = 0;
};

class EncodeFrameProcessor {
 public:
  EncodeFrameProcessor(const EncodeStreamOptions& options, mlvc::StageModelSet* models,
                       const mlvc::RuntimeSidecar* sidecar, mlvc::Profiler* profiler,
                       mlvc::EntropyWorker* entropy_worker, EncodeState* state,
                       MlvcRateController* rate_controller,
                       mlvc::MlvcOfficialEntropyEncoder* entropy_encoder,
                       EncodeDimensions dimensions, double fps, SourceFrameGeometry geometry,
                       aclrtContext context);

  PendingEncodedFrame Process(const std::shared_ptr<mlvc::DataObject>& data,
                              int expected_frame_index);
  void ValidateMotionFrameCount(int frames) const;
  std::size_t motion_nonzero_frames() const { return motion_nonzero_frames_; }

 private:
  const EncodeStreamOptions& options_;
  mlvc::StageModelSet* models_ = nullptr;
  const mlvc::RuntimeSidecar* sidecar_ = nullptr;
  mlvc::Profiler* profiler_ = nullptr;
  mlvc::EntropyWorker* entropy_worker_ = nullptr;
  EncodeState* state_ = nullptr;
  MlvcRateController* rate_controller_ = nullptr;
  mlvc::MlvcOfficialEntropyEncoder* entropy_encoder_ = nullptr;
  EncodeDimensions dimensions_;
  double fps_ = 30.0;
  std::unique_ptr<mlvc::motion::TranslationEstimator> motion_estimator_;
  std::unique_ptr<mlvc::motion::DvppTranslationEstimator> dvpp_motion_estimator_;
  std::vector<mlvc::motion::Translation> replay_shifts_;
  std::size_t motion_nonzero_frames_ = 0;
};

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_ENCODE_ENCODE_FRAME_H_
