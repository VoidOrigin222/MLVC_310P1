#include "mlvc/framework/streaming_pipeline.h"

#include <algorithm>
#include <utility>

#include "mlvc/core/status.h"

namespace mlvc {

StreamingPipeline::StreamingPipeline(std::size_t worker_count,
                                     std::size_t queue_capacity)
    : worker_count_(std::max<std::size_t>(1, worker_count)),
      queue_capacity_(std::max<std::size_t>(1, queue_capacity)),
      input_queue_(queue_capacity_),
      output_queue_(queue_capacity_) {}

StreamingPipeline::~StreamingPipeline() { Stop(); }

void StreamingPipeline::SetProcessor(Processor processor) {
  Check(!running(), "cannot change processor while pipeline is running");
  Check(processor != nullptr, "pipeline processor must not be empty");
  processor_ = std::move(processor);
}

void StreamingPipeline::Start() {
  Check(!running(), "pipeline is already running");
  Check(processor_ != nullptr, "pipeline processor is not configured");
  for (std::thread& thread : processing_threads_) {
    if (thread.joinable()) thread.join();
  }
  processing_threads_.clear();
  input_queue_.Reset();
  output_queue_.Reset();
  processed_count_ = 0;
  error_count_ = 0;
  active_workers_ = worker_count_;
  running_ = true;
  processing_threads_.reserve(worker_count_);
  for (std::size_t index = 0; index < worker_count_; ++index) {
    processing_threads_.emplace_back(&StreamingPipeline::ProcessingLoop, this);
  }
}

void StreamingPipeline::Stop() {
  const bool was_running = running_.exchange(false);
  if (was_running) {
    input_queue_.Close();
    output_queue_.Close();
  }
  for (std::thread& thread : processing_threads_) {
    if (thread.joinable()) thread.join();
  }
  processing_threads_.clear();
  output_queue_.Close();
}

void StreamingPipeline::CloseInput() { input_queue_.Close(); }

bool StreamingPipeline::AddInput(std::shared_ptr<DataObject> input) {
  if (!input || !running()) return false;
  return input_queue_.Push(std::move(input));
}

bool StreamingPipeline::TryGetOutput(std::shared_ptr<DataObject>* output) {
  return output_queue_.TryPop(output);
}

bool StreamingPipeline::GetOutput(std::shared_ptr<DataObject>* output) {
  return output_queue_.Pop(output);
}

void StreamingPipeline::ProcessingLoop() {
  std::shared_ptr<DataObject> input;
  while (input_queue_.Pop(&input)) {
    try {
      auto output = processor_(input);
      if (output && output_queue_.Push(std::move(output))) {
        ++processed_count_;
      } else if (output) {
        break;
      }
    } catch (...) {
      ++error_count_;
    }
  }
  if (active_workers_.fetch_sub(1) == 1) {
    running_ = false;
    output_queue_.Close();
  }
}

}  // namespace mlvc
