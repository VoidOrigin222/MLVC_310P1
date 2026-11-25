#ifndef ULBVC_FRAMEWORK_PROFILE_RANGE_H_
#define ULBVC_FRAMEWORK_PROFILE_RANGE_H_

#include <atomic>
#include <string>
#include <utility>

namespace ulbvc {

inline std::atomic<int> g_profile_hot_path_depth{0};

inline bool ProfileHotPathEnabled() {
  return g_profile_hot_path_depth.load(std::memory_order_relaxed) > 0;
}

class ScopedProfileHotPath {
 public:
  ScopedProfileHotPath() { g_profile_hot_path_depth.fetch_add(1, std::memory_order_relaxed); }

  ScopedProfileHotPath(const ScopedProfileHotPath&) = delete;
  ScopedProfileHotPath& operator=(const ScopedProfileHotPath&) = delete;

  ~ScopedProfileHotPath() { g_profile_hot_path_depth.fetch_sub(1, std::memory_order_relaxed); }
};

class ScopedProfileRange {
 public:
  explicit ScopedProfileRange(const char* name) : name_(name), active_(ProfileHotPathEnabled()) {}

  explicit ScopedProfileRange(std::string name)
      : name_(std::move(name)), active_(ProfileHotPathEnabled()) {}

  ScopedProfileRange(const ScopedProfileRange&) = delete;
  ScopedProfileRange& operator=(const ScopedProfileRange&) = delete;

  ~ScopedProfileRange() = default;

 private:
  std::string name_;
  bool active_ = false;
};

}  // namespace ulbvc

#define ULBVC_PROFILE_JOIN_INNER(a, b) a##b
#define ULBVC_PROFILE_JOIN(a, b) ULBVC_PROFILE_JOIN_INNER(a, b)
#define ULBVC_PROFILE_RANGE_FUNCTION() \
  ::ulbvc::ScopedProfileRange ULBVC_PROFILE_JOIN(ulbvc_profile_range_, __LINE__)(__func__)
#define ULBVC_PROFILE_RANGE(name) \
  ::ulbvc::ScopedProfileRange ULBVC_PROFILE_JOIN(ulbvc_profile_range_, __LINE__)(name)

#endif  // ULBVC_FRAMEWORK_PROFILE_RANGE_H_
