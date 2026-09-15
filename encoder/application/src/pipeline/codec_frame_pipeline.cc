#include <mlvc/application/pipeline/codec_frame_pipeline.h>

#include "mlvc/core/status.h"

namespace mlvc::app {

void PreparedFramePacket::Release() {
  if (queue_ == nullptr || frame_.slot_index < 0) return;
  queue_->Release(&frame_);
  queue_ = nullptr;
}

void PreparedFrameProducer::Produce() {
  while (ShouldContinue()) {
    InputFrame frame = input_queue_.Pop();
    if (frame.eof) break;
    Check(frame.frame != nullptr, "prepared frame producer received an empty frame");
    if (!AddData(std::make_shared<PreparedFramePacket>(&input_queue_, frame))) break;
  }
  pipeline_.CloseInput();
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
  while (pipeline_.GetOutput(&data)) {
    consume_function_(data);
  }
}

}  // namespace mlvc::app
