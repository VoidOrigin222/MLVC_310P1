#include "mlvc/framework/data_producer.h"

#include "mlvc/framework/streaming_pipeline.h"

namespace mlvc {

DataProducer::DataProducer(StreamingPipeline& pipeline,
                           std::atomic<bool>& running)
    : pipeline_(pipeline), running_(running) {}

DataProducer::~DataProducer() {
  Stop();
  Join();
}

void DataProducer::Start() {
  if (thread_.joinable()) return;
  running_ = true;
  thread_ = std::thread(&DataProducer::RunThread, this);
}

void DataProducer::Stop() {
  running_ = false;
  pipeline_.Stop();
}

void DataProducer::Join() {
  if (thread_.joinable()) thread_.join();
}

void DataProducer::RethrowIfFailed() const {
  std::lock_guard<std::mutex> lock(error_mutex_);
  if (error_) std::rethrow_exception(error_);
}

void DataProducer::RunThread() {
  try {
    Produce();
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_ = std::current_exception();
    }
    running_ = false;
    pipeline_.CloseInput();
  }
}

bool DataProducer::AddData(std::shared_ptr<DataObject> data) {
  return pipeline_.AddInput(std::move(data));
}

}  // namespace mlvc
