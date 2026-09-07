#ifndef MLVC_FRAMEWORK_STREAMING_PIPELINE_H_
#define MLVC_FRAMEWORK_STREAMING_PIPELINE_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "mlvc/framework/data_object.h"

namespace mlvc {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  bool Push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [this] { return closed_ || queue_.size() < capacity_; });
    if (closed_) return false;
    queue_.push(std::move(value));
    not_empty_.notify_one();
    return true;
  }

  bool Pop(T* value) {
    if (value == nullptr) return false;
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) return false;
    *value = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
  }

  bool TryPop(T* value) {
    if (value == nullptr) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    *value = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    not_full_.notify_all();
    not_empty_.notify_all();
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<T> empty;
    queue_.swap(empty);
    closed_ = false;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  std::queue<T> queue_;
  bool closed_ = false;
};

class StreamingPipeline {
 public:
  using Processor =
      std::function<std::shared_ptr<DataObject>(const std::shared_ptr<DataObject>&)>;

  explicit StreamingPipeline(std::size_t worker_count = 0,
                             std::size_t queue_capacity = 8);
  ~StreamingPipeline();

  StreamingPipeline(const StreamingPipeline&) = delete;
  StreamingPipeline& operator=(const StreamingPipeline&) = delete;

  void SetProcessor(Processor processor);
  void Start();
  void Stop();
  void CloseInput();
  bool AddInput(std::shared_ptr<DataObject> input);
  bool TryGetOutput(std::shared_ptr<DataObject>* output);
  bool GetOutput(std::shared_ptr<DataObject>* output);
  bool running() const { return running_.load(); }
  std::size_t processed_count() const { return processed_count_.load(); }
  std::size_t error_count() const { return error_count_.load(); }

 private:
  void ProcessingLoop();

  const std::size_t worker_count_;
  const std::size_t queue_capacity_;
  BoundedQueue<std::shared_ptr<DataObject>> input_queue_;
  BoundedQueue<std::shared_ptr<DataObject>> output_queue_;
  Processor processor_;
  std::vector<std::thread> processing_threads_;
  std::atomic<std::size_t> active_workers_{0};
  std::atomic<bool> running_{false};
  std::atomic<std::size_t> processed_count_{0};
  std::atomic<std::size_t> error_count_{0};
};

}  // namespace mlvc

#endif  // MLVC_FRAMEWORK_STREAMING_PIPELINE_H_
