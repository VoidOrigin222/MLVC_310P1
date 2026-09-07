#ifndef MLVC_FRAMEWORK_ENTROPY_WORKER_H_
#define MLVC_FRAMEWORK_ENTROPY_WORKER_H_

#include <cstdint>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <vector>

#include "mlvc/framework/thread_pool.h"

namespace mlvc {

class EntropyWorker final : public TaskExecutor {
 public:
  explicit EntropyWorker(std::size_t thread_count = 1);
  ~EntropyWorker();

  EntropyWorker(const EntropyWorker&) = delete;
  EntropyWorker& operator=(const EntropyWorker&) = delete;

  std::future<std::vector<uint8_t>> Submit(std::function<std::vector<uint8_t>()> task);
  std::future<void> SubmitVoid(std::function<void()> task);
  std::future<void> SubmitTask(std::unique_ptr<Task> task) override;

  // Exchange typed frame jobs with the NPU-driving thread. Callers must give
  // concurrently submitted tasks independent mutable codec state.
  template <typename T>
  std::future<T> SubmitValue(std::function<T()> task) {
    return worker_pool_.SubmitValue<T>(std::move(task));
  }
  void Stop() override;

 private:
  ThreadPool worker_pool_;
};

}  // namespace mlvc

#endif  // MLVC_FRAMEWORK_ENTROPY_WORKER_H_
