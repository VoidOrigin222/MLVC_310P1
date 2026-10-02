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

#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>

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

  ~EncodeFrameProcessor();

  PendingEncodedFrame Process(const std::shared_ptr<mlvc::DataObject>& data,
                              int expected_frame_index);
  void ValidateMotionFrameCount(int frames) const;
  void PrepareMotion(const std::shared_ptr<mlvc::DataObject>& data);
  const LatencyStats& motion_work() const { return motion_work_; }
  const LatencyStats& ready_to_motion() const { return ready_to_motion_; }
  const LatencyStats& ready_to_encode() const { return ready_to_encode_; }
  std::size_t motion_nonzero_frames() const { return motion_nonzero_frames_; }

 private:
  mlvc::motion::Translation EstimateMotion(const mlvc::app::InputFrame& frame, bool random_access);
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
  aclrtContext context_ = nullptr;
  mlvc::io::Nv12Layout motion_layout_;
  int next_motion_frame_ = 0;
  struct EntropyTasks {
    std::mutex mutex;
    std::condition_variable condition;
    std::size_t active = 0;
    void Complete() {
      std::lock_guard<std::mutex> lock(mutex);
      --active;
      condition.notify_all();
    }
  };
  EntropyTasks entropy_tasks_;
  TensorData warped_reference_;
  LatencyStats motion_work_;
  LatencyStats ready_to_motion_;
  LatencyStats ready_to_encode_;
};

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_ENCODE_ENCODE_FRAME_H_
