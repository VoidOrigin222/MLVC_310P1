#include "mlvc/framework/streaming_pipeline.h"

#include <algorithm>
#include <utility>

#include "mlvc/core/status.h"

namespace mlvc {

StreamingPipeline::StreamingPipeline(std::size_t worker_count, std::size_t queue_capacity)
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
  next_sequence_ = 0;
  {
    std::lock_guard<std::mutex> lock(reorder_mutex_);
    completed_results_.clear();
    next_output_sequence_ = 0;
    first_error_ = nullptr;
  }
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
  const std::size_t sequence = next_sequence_.fetch_add(1);
  return input_queue_.Push(WorkItem{sequence, std::move(input)});
}

bool StreamingPipeline::TryGetOutput(std::shared_ptr<DataObject>* output) {
  return output_queue_.TryPop(output);
}

bool StreamingPipeline::GetOutput(std::shared_ptr<DataObject>* output) {
  return output_queue_.Pop(output);
}

void StreamingPipeline::RethrowIfFailed() const {
  std::lock_guard<std::mutex> lock(reorder_mutex_);
  if (first_error_) std::rethrow_exception(first_error_);
}

void StreamingPipeline::ProcessingLoop() {
  WorkItem work;
  while (input_queue_.Pop(&work)) {
    std::shared_ptr<DataObject> output;
    try {
      output = processor_(work.input);
    } catch (...) {
      ++error_count_;
      std::lock_guard<std::mutex> lock(reorder_mutex_);
      if (!first_error_) first_error_ = std::current_exception();
    }
    {
      std::unique_lock<std::mutex> lock(reorder_mutex_);
      completed_results_.emplace(work.sequence, std::move(output));
      PublishReadyResults();
    }
  }
  if (active_workers_.fetch_sub(1) == 1) {
    running_ = false;
    output_queue_.Close();
  }
}

void StreamingPipeline::PublishReadyResults() {
  while (true) {
    auto it = completed_results_.find(next_output_sequence_);
    if (it == completed_results_.end()) return;
    std::shared_ptr<DataObject> output = std::move(it->second);
    completed_results_.erase(it);
    ++next_output_sequence_;
    if (output) {
      if (output_queue_.Push(std::move(output))) {
        ++processed_count_;
      } else {
        return;
      }
    }
  }
}

}  // namespace mlvc
