#ifndef MLVC_APPLICATION_PIPELINE_CODEC_FRAME_PIPELINE_H_
#define MLVC_APPLICATION_PIPELINE_CODEC_FRAME_PIPELINE_H_

#include <mlvc/application/input/frame_input_queue.h>
#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/framework/data_consumer.h>
#include <mlvc/framework/data_object.h>
#include <mlvc/framework/data_producer.h>
#include <mlvc/framework/streaming_pipeline.h>
#include <mlvc/io/mlvc_bitstream.h>
#include <mlvc/motion/translation_estimator.h>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace mlvc::app {

// A frame packet owns the input-queue slot until the codec consumer releases it.
// The packet is deliberately a DataObject so the common pipeline stays codec-agnostic.
class PreparedFramePacket final : public mlvc::DataObject {
 public:
  PreparedFramePacket(AsyncFrameInputQueue* queue, InputFrame frame)
      : queue_(queue), frame_(frame) {}
  ~PreparedFramePacket() override { Release(); }

  PreparedFramePacket(const PreparedFramePacket&) = delete;
  PreparedFramePacket& operator=(const PreparedFramePacket&) = delete;

  InputFrame& frame() { return frame_; }
  const InputFrame& frame() const { return frame_; }
  void Release();
  std::optional<mlvc::motion::Translation> motion;
  std::chrono::steady_clock::time_point motion_ready_at{};

 private:
  AsyncFrameInputQueue* queue_ = nullptr;
  InputFrame frame_;
};

// Reads prepared frames from the asynchronous input queue and applies backpressure
// through StreamingPipeline's bounded input queue.
class PreparedFrameProducer final : public mlvc::DataProducer {
 public:
  PreparedFrameProducer(mlvc::StreamingPipeline& pipeline, std::atomic<bool>& running,
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

struct BitstreamPacket final : public mlvc::DataObject {
  int frame_index = 0;
  mlvc::codec::MlvcFrameType frame_type = mlvc::codec::MlvcFrameType::kPFrame;
  int q_index = 0;
  mlvc::io::MlvcFrameMetadata metadata;
  std::vector<uint8_t> payload;
};

class BitstreamProducer final : public mlvc::DataProducer {
 public:
  using ReadFunction = std::function<std::shared_ptr<BitstreamPacket>()>;

  BitstreamProducer(mlvc::StreamingPipeline& pipeline, std::atomic<bool>& running,
                    ReadFunction read_function)
      : DataProducer(pipeline, running), read_function_(std::move(read_function)) {}

 protected:
  void Produce() override;

 private:
  ReadFunction read_function_;
};

class CallbackDataConsumer final : public mlvc::DataConsumer {
 public:
  using ConsumeFunction = std::function<void(const std::shared_ptr<mlvc::DataObject>&)>;

  CallbackDataConsumer(mlvc::StreamingPipeline& pipeline, std::atomic<bool>& running,
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

}  // namespace mlvc::app

#endif  // MLVC_APPLICATION_PIPELINE_CODEC_FRAME_PIPELINE_H_
