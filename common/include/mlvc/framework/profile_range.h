#ifndef MLVC_FRAMEWORK_PROFILE_RANGE_H_
#define MLVC_FRAMEWORK_PROFILE_RANGE_H_

#include <atomic>
#include <string>
#include <utility>

namespace mlvc {

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

}  // namespace mlvc

#define MLVC_PROFILE_JOIN_INNER(a, b) a##b
#define MLVC_PROFILE_JOIN(a, b) MLVC_PROFILE_JOIN_INNER(a, b)
#define MLVC_PROFILE_RANGE_FUNCTION() \
  ::mlvc::ScopedProfileRange MLVC_PROFILE_JOIN(mlvc_profile_range_, __LINE__)(__func__)
#define MLVC_PROFILE_RANGE(name) \
  ::mlvc::ScopedProfileRange MLVC_PROFILE_JOIN(mlvc_profile_range_, __LINE__)(name)

#endif  // MLVC_FRAMEWORK_PROFILE_RANGE_H_
