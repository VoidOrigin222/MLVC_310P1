#include <mlvc/codec/detail/profile/allocation_tracking.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace mlvc::codec {

std::atomic<bool> g_track_allocations{false};
std::atomic<bool> g_track_repository_allocations{false};
std::atomic<uint64_t> g_allocation_count{0};
std::atomic<uint64_t> g_allocation_bytes{0};
std::atomic<uint64_t> g_repository_allocation_count{0};
std::atomic<uint64_t> g_repository_allocation_bytes{0};

}  // namespace mlvc::codec

namespace {

void TrackAllocation(std::size_t bytes) {
  if (mlvc::codec::g_track_allocations.load(std::memory_order_relaxed)) {
    mlvc::codec::g_allocation_count.fetch_add(1, std::memory_order_relaxed);
    mlvc::codec::g_allocation_bytes.fetch_add(static_cast<uint64_t>(bytes),
                                              std::memory_order_relaxed);
  }
  if (mlvc::codec::g_track_repository_allocations.load(std::memory_order_relaxed)) {
    mlvc::codec::g_repository_allocation_count.fetch_add(1, std::memory_order_relaxed);
    mlvc::codec::g_repository_allocation_bytes.fetch_add(static_cast<uint64_t>(bytes),
                                                         std::memory_order_relaxed);
  }
}

}  // namespace

void* operator new(std::size_t bytes) {
  TrackAllocation(bytes);
  if (void* pointer = std::malloc(bytes)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t bytes) {
  TrackAllocation(bytes);
  if (void* pointer = std::malloc(bytes)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
  TrackAllocation(bytes);
  return std::malloc(bytes);
}

void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
  TrackAllocation(bytes);
  return std::malloc(bytes);
}

void* operator new(std::size_t bytes, std::align_val_t alignment) {
  TrackAllocation(bytes);
  void* pointer = nullptr;
  const std::size_t alignment_value = static_cast<std::size_t>(alignment);
  const int status = posix_memalign(&pointer, alignment_value, bytes);
  if (status == 0) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t bytes, std::align_val_t alignment) {
  TrackAllocation(bytes);
  void* pointer = nullptr;
  const std::size_t alignment_value = static_cast<std::size_t>(alignment);
  const int status = posix_memalign(&pointer, alignment_value, bytes);
  if (status == 0) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new(std::size_t bytes, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  TrackAllocation(bytes);
  void* pointer = nullptr;
  const std::size_t alignment_value = static_cast<std::size_t>(alignment);
  if (posix_memalign(&pointer, alignment_value, bytes) != 0) {
    return nullptr;
  }
  return pointer;
}

void* operator new[](std::size_t bytes, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
  TrackAllocation(bytes);
  void* pointer = nullptr;
  const std::size_t alignment_value = static_cast<std::size_t>(alignment);
  if (posix_memalign(&pointer, alignment_value, bytes) != 0) {
    return nullptr;
  }
  return pointer;
}

void operator delete(void* pointer) noexcept { std::free(pointer); }

void operator delete[](void* pointer) noexcept { std::free(pointer); }

void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

void operator delete(void* pointer, const std::nothrow_t&) noexcept { std::free(pointer); }

void operator delete[](void* pointer, const std::nothrow_t&) noexcept { std::free(pointer); }

void operator delete(void* pointer, std::align_val_t) noexcept { std::free(pointer); }

void operator delete[](void* pointer, std::align_val_t) noexcept { std::free(pointer); }

void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }

void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
