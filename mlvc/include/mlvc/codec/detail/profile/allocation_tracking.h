#ifndef MLVC_CODEC_DETAIL_ALLOCATION_TRACKING_H_
#define MLVC_CODEC_DETAIL_ALLOCATION_TRACKING_H_

#include <atomic>
#include <cstdint>

namespace mlvc::codec {

extern std::atomic<bool> g_track_allocations;
extern std::atomic<bool> g_track_repository_allocations;
extern std::atomic<uint64_t> g_allocation_count;
extern std::atomic<uint64_t> g_allocation_bytes;
extern std::atomic<uint64_t> g_repository_allocation_count;
extern std::atomic<uint64_t> g_repository_allocation_bytes;

struct AllocationSnapshot {
  uint64_t count = 0;
  uint64_t bytes = 0;
};

AllocationSnapshot SnapshotAllocations();
AllocationSnapshot SnapshotFullProcessAllocations();
bool AllocationTrackingEnabled();
bool RepositoryAllocationTrackingEnabled();
void SetAllocationTracking(bool enabled);
void SetRepositoryAllocationTracking(bool enabled);

class ScopedRepositoryAllocationTrackingPause {
 public:
  ScopedRepositoryAllocationTrackingPause();

  ScopedRepositoryAllocationTrackingPause(const ScopedRepositoryAllocationTrackingPause&) = delete;
  ScopedRepositoryAllocationTrackingPause& operator=(
      const ScopedRepositoryAllocationTrackingPause&) = delete;

  ~ScopedRepositoryAllocationTrackingPause();

 private:
  bool was_repository_tracking_ = false;
};

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_ALLOCATION_TRACKING_H_
