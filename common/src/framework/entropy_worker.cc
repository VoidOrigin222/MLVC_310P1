#include "mlvc/framework/entropy_worker.h"

#include <utility>

namespace mlvc {

EntropyWorker::EntropyWorker(std::size_t thread_count) : worker_pool_(thread_count) {}

EntropyWorker::~EntropyWorker() { Stop(); }

std::future<void> EntropyWorker::SubmitTask(std::unique_ptr<Task> task) {
  return worker_pool_.SubmitTask(std::move(task));
}

std::future<std::vector<uint8_t>> EntropyWorker::Submit(
    std::function<std::vector<uint8_t>()> task) {
  return worker_pool_.SubmitValue<std::vector<uint8_t>>(std::move(task));
}

std::future<void> EntropyWorker::SubmitVoid(std::function<void()> task) {
  mlvc::Check(task != nullptr, "cannot submit an empty entropy task");
  return SubmitTask(std::make_unique<FunctionTask>("function", -1, std::move(task)));
}

void EntropyWorker::Stop() { worker_pool_.Stop(); }

}  // namespace mlvc
