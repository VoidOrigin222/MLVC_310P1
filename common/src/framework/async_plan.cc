#include "mlvc/framework/async_plan.h"

#include "mlvc/core/status.h"

namespace mlvc {

AsyncFramePlan::AsyncFramePlan(std::size_t pinned_bytes_per_slot, int slots) {
  Check(slots > 0, "async plan requires at least one pinned slot");
  pinned_slots_.reserve(static_cast<std::size_t>(slots));
  for (int i = 0; i < slots; ++i) {
    pinned_slots_.emplace_back(pinned_bytes_per_slot);
  }
  copy_records_.reserve(8);
}

PinnedHostBuffer& AsyncFramePlan::PinnedSlot(int index) {
  Check(index >= 0 && index < static_cast<int>(pinned_slots_.size()), "invalid pinned slot index");
  return pinned_slots_[index];
}

void AsyncFramePlan::RecordCopy(const std::string& name, std::size_t bytes) {
  RecordCopy(name, bytes, 0.0, 0.0, "host");
}

void AsyncFramePlan::RecordCopy(const std::string& name, std::size_t bytes, double start_ms,
                                double duration_ms, const std::string& timing_source) {
  copy_records_.push_back(AsyncCopyRecord{name, bytes, start_ms, duration_ms, timing_source});
}

void AsyncFramePlan::ClearCopyRecords() { copy_records_.clear(); }

}  // namespace mlvc
