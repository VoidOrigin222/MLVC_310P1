#ifndef ULBVC_APPLICATION_PIPELINE_CODEC_FRAME_PIPELINE_H_
#define ULBVC_APPLICATION_PIPELINE_CODEC_FRAME_PIPELINE_H_

#include <ulbvc/application/input/frame_input_queue.h>
#include <ulbvc/codec/ulbvc_entropy.h>
#include <ulbvc/framework/data_consumer.h>
#include <ulbvc/framework/data_object.h>
#include <ulbvc/framework/data_producer.h>
#include <ulbvc/framework/streaming_pipeline.h>
#include <ulbvc/io/ulbvc_bitstream.h>
#include <ulbvc/motion/translation_estimator.h>
#include <ulbvc/transport/detection_metadata.h>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace ulbvc::app {

// A frame packet owns the input-queue slot until the codec consumer releases it.
// The packet is deliberately a DataObject so the common pipeline stays codec-agnostic.
class PreparedFramePacket final : public ulbvc::DataObject {
 public:
  PreparedFramePacket(AsyncFrameInputQueue* queue, InputFrame frame)
      : queue_(queue), frame_(frame) {}
  ~PreparedFramePacket() override { Release(); }

  PreparedFramePacket(const PreparedFramePacket&) = delete;
  PreparedFramePacket& operator=(const PreparedFramePacket&) = delete;

  InputFrame& frame() { return frame_; }
  const InputFrame& frame() const { return frame_; }
  void Release();
  std::optional<ulbvc::motion::Translation> motion;
  bool force_random_access = false;
  std::chrono::steady_clock::time_point motion_ready_at{};
  std::shared_future<ulbvc::transport::DetectionFrame> detection;

 private:
  AsyncFrameInputQueue* queue_ = nullptr;
  InputFrame frame_;
};

// Reads prepared frames from the asynchronous input queue and applies backpressure
// through StreamingPipeline's bounded input queue.
class PreparedFrameProducer final : public ulbvc::DataProducer {
 public:
  PreparedFrameProducer(ulbvc::StreamingPipeline& pipeline, std::atomic<bool>& running,
                        AsyncFrameInputQueue& input_queue)
      : DataProducer(pipeline, running), input_queue_(input_queue) {}

  ~PreparedFrameProducer() override {
    input_queue_.Cancel();
    Stop();
    Join();
  }

 protected:
  void Produce() override;

 private:
  AsyncFrameInputQueue& input_queue_;
};

struct BitstreamPacket final : public ulbvc::DataObject {
  int frame_index = 0;
  ulbvc::codec::UlbvcFrameType frame_type = ulbvc::codec::UlbvcFrameType::kPFrame;
  int q_index = 0;
  ulbvc::io::UlbvcFrameMetadata metadata;
  std::vector<uint8_t> payload;
};

class BitstreamProducer final : public ulbvc::DataProducer {
 public:
  using ReadFunction = std::function<std::shared_ptr<BitstreamPacket>()>;

  BitstreamProducer(ulbvc::StreamingPipeline& pipeline, std::atomic<bool>& running,
                    ReadFunction read_function)
      : DataProducer(pipeline, running), read_function_(std::move(read_function)) {}

 protected:
  void Produce() override;

 private:
  ReadFunction read_function_;
};

class CallbackDataConsumer final : public ulbvc::DataConsumer {
 public:
  using ConsumeFunction = std::function<void(const std::shared_ptr<ulbvc::DataObject>&)>;

  CallbackDataConsumer(ulbvc::StreamingPipeline& pipeline, std::atomic<bool>& running,
                       ConsumeFunction consume_function, std::function<void()> cancel = {})
      : DataConsumer(pipeline, running), consume_function_(std::move(consume_function)),
        cancel_(std::move(cancel)) {}

  double wait_ms() const { return wait_ms_; }
  double wait_max_ms() const { return wait_max_ms_; }

  ~CallbackDataConsumer() override {
    Stop();
    Join();
  }

 protected:
  void Consume() override;

 private:
  ConsumeFunction consume_function_;
  std::function<void()> cancel_;
  double wait_ms_ = 0.0;
  double wait_max_ms_ = 0.0;
};

}  // namespace ulbvc::app

#endif  // ULBVC_APPLICATION_PIPELINE_CODEC_FRAME_PIPELINE_H_
