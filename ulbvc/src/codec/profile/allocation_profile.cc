#include <ulbvc/codec/detail/profile/allocation_profile.h>

namespace ulbvc::codec {
namespace codec = ulbvc::codec;

void AddAllocationProfile(ulbvc::Profiler* profiler, const char* name,
                          const codec::AllocationSnapshot& before,
                          const codec::AllocationSnapshot& after) {
  if (profiler == nullptr) {
    return;
  }
  const bool was_tracking = codec::AllocationTrackingEnabled();
  codec::SetAllocationTracking(false);
  profiler->AddCounter(
      "allocation." + std::string(name) + ".count", "allocation", "allocation", 0.0,
      {ulbvc::Profiler::Arg("count", static_cast<uint64_t>(after.count - before.count))});
  profiler->AddCounter(
      "allocation." + std::string(name) + ".bytes", "allocation", "allocation", 0.0,
      {ulbvc::Profiler::Arg("bytes", static_cast<uint64_t>(after.bytes - before.bytes))});
  codec::SetAllocationTracking(was_tracking);
}

void AddHotPathAllocationProfile(ulbvc::Profiler* profiler, const codec::AllocationSnapshot& before,
                                 const codec::AllocationSnapshot& after) {
  AddAllocationProfile(profiler, "hot_path", before, after);
}

ScopedAllocationSpan::ScopedAllocationSpan(ulbvc::Profiler* profiler, const char* name)
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

}  // namespace ulbvc::codec
