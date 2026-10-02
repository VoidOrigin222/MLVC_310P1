#include <mlvc/application/pipeline/codec_frame_pipeline.h>

#include <algorithm>

#include "mlvc/core/status.h"

namespace mlvc::app {

void PreparedFramePacket::Release() {
  if (queue_ == nullptr || frame_.slot_index < 0) return;
  queue_->Release(&frame_);
  queue_ = nullptr;
}

void PreparedFrameProducer::Produce() {
  try {
    while (ShouldContinue()) {
      InputFrame frame = input_queue_.Pop();
      if (frame.eof) break;
      auto packet = std::make_shared<PreparedFramePacket>(&input_queue_, frame);
      Check(frame.frame != nullptr, "prepared frame producer received an empty frame");
      if (!AddData(std::move(packet))) break;
    }
    pipeline_.CloseInput();
  } catch (...) {
    input_queue_.Cancel(std::current_exception());
    throw;
  }
}

void BitstreamProducer::Produce() {
  Check(read_function_ != nullptr, "bitstream producer read function is empty");
  while (ShouldContinue()) {
    std::shared_ptr<BitstreamPacket> packet = read_function_();
    if (!packet || !AddData(std::move(packet))) break;
  }
  pipeline_.CloseInput();
}

void CallbackDataConsumer::Consume() {
  Check(consume_function_ != nullptr, "data consumer callback is empty");
  std::shared_ptr<mlvc::DataObject> data;
  try {
    while (true) {
      const auto begin = std::chrono::steady_clock::now();
      const bool received = pipeline_.GetOutput(&data);
      const double elapsed = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - begin).count();
      wait_ms_ += elapsed;
      wait_max_ms_ = std::max(wait_max_ms_, elapsed);
      if (!received) break;
      consume_function_(data);
    }
  } catch (...) {
    if (cancel_) cancel_();
    throw;
  }
}

}  // namespace mlvc::app
