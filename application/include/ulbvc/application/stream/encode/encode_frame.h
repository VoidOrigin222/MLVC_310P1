#ifndef ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_FRAME_H_
#define ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_FRAME_H_

#include <ulbvc/application/pipeline/codec_frame_pipeline.h>
#include <ulbvc/application/stream/encode/encode_state.h>
#include <ulbvc/application/stream/ulbvc_internal.h>
#include <ulbvc/codec/ulbvc_rate_control.h>
#include <ulbvc/entropy/ulbvc_official_entropy.h>
#include <ulbvc/entropy/sidecar.h>
#include <ulbvc/framework/codec_graph_executor.h>
#include <ulbvc/framework/entropy_worker.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/runtime/stage_runtime.h>
#include <ulbvc/motion/translation_estimator.h>
#include <ulbvc/motion/dvpp_translation_estimator.h>

#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>

namespace ulbvc::codec {

struct EncodeDimensions {
  int y_channels = 0;
  int y_height = 0;
  int y_width = 0;
  int z_height = 0;
  int z_width = 0;
};

class EncodeFrameProcessor {
 public:
  EncodeFrameProcessor(const EncodeStreamOptions& options, ulbvc::StageModelSet* models,
                       const ulbvc::RuntimeSidecar* sidecar, ulbvc::Profiler* profiler,
                       ulbvc::EntropyWorker* entropy_worker, EncodeState* state,
                       UlbvcRateController* rate_controller,
                       ulbvc::UlbvcOfficialEntropyEncoder* entropy_encoder,
                       EncodeDimensions dimensions, double fps, SourceFrameGeometry geometry,
                       aclrtContext context);

  ~EncodeFrameProcessor();

  PendingEncodedFrame Process(const std::shared_ptr<ulbvc::DataObject>& data,
                              int expected_frame_index);
  void ValidateMotionFrameCount(int frames) const;
  void PrepareMotion(const std::shared_ptr<ulbvc::DataObject>& data);
  const LatencyStats& motion_work() const { return motion_work_; }
  const LatencyStats& ready_to_motion() const { return ready_to_motion_; }
  const LatencyStats& ready_to_encode() const { return ready_to_encode_; }
  std::size_t motion_nonzero_frames() const { return motion_nonzero_frames_; }

 private:
  ulbvc::motion::Translation EstimateMotion(const ulbvc::app::InputFrame& frame, bool random_access);
  const EncodeStreamOptions& options_;
  ulbvc::StageModelSet* models_ = nullptr;
  const ulbvc::RuntimeSidecar* sidecar_ = nullptr;
  ulbvc::Profiler* profiler_ = nullptr;
  ulbvc::EntropyWorker* entropy_worker_ = nullptr;
  EncodeState* state_ = nullptr;
  UlbvcRateController* rate_controller_ = nullptr;
  ulbvc::UlbvcOfficialEntropyEncoder* entropy_encoder_ = nullptr;
  EncodeDimensions dimensions_;
  double fps_ = 30.0;
  std::unique_ptr<ulbvc::motion::TranslationEstimator> motion_estimator_;
  std::unique_ptr<ulbvc::motion::DvppTranslationEstimator> dvpp_motion_estimator_;
  std::vector<ulbvc::motion::Translation> replay_shifts_;
  std::size_t motion_nonzero_frames_ = 0;
  aclrtContext context_ = nullptr;
  ulbvc::io::Nv12Layout motion_layout_;
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

}  // namespace ulbvc::codec

#endif  // ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_FRAME_H_
