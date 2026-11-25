#ifndef ULBVC_FRAMEWORK_ASYNC_PLAN_H_
#define ULBVC_FRAMEWORK_ASYNC_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ulbvc/core/buffer.h"
#include "ulbvc/framework/entropy_worker.h"

namespace ulbvc {

struct AsyncCopyRecord {
  std::string name;
  std::size_t bytes = 0;
  double start_ms = 0.0;
  double duration_ms = 0.0;
  std::string timing_source = "host";
};

class AsyncFramePlan {
 public:
  AsyncFramePlan(std::size_t pinned_bytes_per_slot, int slots);

  PinnedHostBuffer& PinnedSlot(int index);
  EntropyWorker& entropy_worker() { return entropy_worker_; }

  void RecordCopy(const std::string& name, std::size_t bytes);
  void RecordCopy(const std::string& name, std::size_t bytes, double start_ms, double duration_ms,
                  const std::string& timing_source);
  void ClearCopyRecords();
  const std::vector<AsyncCopyRecord>& copy_records() const { return copy_records_; }

 private:
  std::vector<PinnedHostBuffer> pinned_slots_;
  EntropyWorker entropy_worker_;
  std::vector<AsyncCopyRecord> copy_records_;
};

}  // namespace ulbvc

#endif  // ULBVC_FRAMEWORK_ASYNC_PLAN_H_
