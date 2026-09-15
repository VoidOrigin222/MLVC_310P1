#include <mlvc/codec/detail/profile/allocation_profile.h>

namespace mlvc::codec {
namespace codec = mlvc::codec;

void AddAllocationProfile(mlvc::Profiler* profiler, const char* name,
                          const codec::AllocationSnapshot& before,
                          const codec::AllocationSnapshot& after) {
  if (profiler == nullptr) {
    return;
  }
  const bool was_tracking = codec::AllocationTrackingEnabled();
  codec::SetAllocationTracking(false);
  profiler->AddCounter(
      "allocation." + std::string(name) + ".count", "allocation", "allocation", 0.0,
      {mlvc::Profiler::Arg("count", static_cast<uint64_t>(after.count - before.count))});
  profiler->AddCounter(
      "allocation." + std::string(name) + ".bytes", "allocation", "allocation", 0.0,
      {mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(after.bytes - before.bytes))});
  codec::SetAllocationTracking(was_tracking);
}

void AddHotPathAllocationProfile(mlvc::Profiler* profiler, const codec::AllocationSnapshot& before,
                                 const codec::AllocationSnapshot& after) {
  AddAllocationProfile(profiler, "hot_path", before, after);
}

ScopedAllocationSpan::ScopedAllocationSpan(mlvc::Profiler* profiler, const char* name)
    : profiler_(profiler), name_(name), before_(codec::SnapshotAllocations()) {}

ScopedAllocationSpan::~ScopedAllocationSpan() {
  AddAllocationProfile(profiler_, name_, before_, codec::SnapshotAllocations());
}

ScopedAllocationTrackingPause::ScopedAllocationTrackingPause()
    : was_tracking_(codec::AllocationTrackingEnabled()),
      was_repository_tracking_(codec::RepositoryAllocationTrackingEnabled()) {
  codec::SetAllocationTracking(false);
}

ScopedAllocationTrackingPause::~ScopedAllocationTrackingPause() {
  codec::g_track_allocations.store(was_tracking_, std::memory_order_relaxed);
  codec::g_track_repository_allocations.store(was_repository_tracking_, std::memory_order_relaxed);
}

}  // namespace mlvc::codec
