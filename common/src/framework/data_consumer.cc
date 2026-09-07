#include "mlvc/framework/data_consumer.h"

#include "mlvc/framework/streaming_pipeline.h"

namespace mlvc {

DataConsumer::DataConsumer(StreamingPipeline& pipeline,
                           std::atomic<bool>& running)
    : pipeline_(pipeline), running_(running) {}

DataConsumer::~DataConsumer() {
  Stop();
  Join();
}

void DataConsumer::Start() {
  if (thread_.joinable()) return;
  running_ = true;
  thread_ = std::thread(&DataConsumer::RunThread, this);
}

void DataConsumer::Stop() {
  running_ = false;
  pipeline_.Stop();
}

void DataConsumer::Join() {
  if (thread_.joinable()) thread_.join();
}

void DataConsumer::RethrowIfFailed() const {
  std::lock_guard<std::mutex> lock(error_mutex_);
  if (error_) std::rethrow_exception(error_);
}

void DataConsumer::RunThread() {
  try {
    Consume();
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_ = std::current_exception();
    }
    running_ = false;
    pipeline_.Stop();
  }
}

bool DataConsumer::GetData(std::shared_ptr<DataObject>* data) {
  return pipeline_.TryGetOutput(data);
}

}  // namespace mlvc
