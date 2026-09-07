#ifndef MLVC_CORE_TENSOR_HANDLE_H_
#define MLVC_CORE_TENSOR_HANDLE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mlvc/core/buffer.h"
#include "mlvc/core/tensor.h"

namespace mlvc {

enum class TensorResidency {
  kEmpty,
  kCpu,
  kPinnedCpu,
  kAcl,
  kCpuAndAcl,
  kPinnedCpuAndAcl,
};

struct TensorCopyResult {
  bool copied = false;
  std::size_t bytes = 0;
  MemoryLocation source = MemoryLocation::kCpu;
  MemoryLocation destination = MemoryLocation::kAcl;
};

struct TensorHandleState {
  std::vector<int64_t> shape;
  DataType dtype = DataType::kFloat16;
  TensorResidency residency = TensorResidency::kEmpty;
  std::size_t bytes = 0;
  bool has_cpu_buffer = false;
  bool has_pinned_cpu_buffer = false;
  bool has_acl_buffer = false;
  bool cpu_valid = false;
  bool pinned_cpu_valid = false;
  bool acl_valid = false;
  bool cpu_dirty = false;
  bool pinned_cpu_dirty = false;
  bool acl_dirty = false;
  bool ready_event_recorded = false;
  std::string last_materialization_reason;
};

class TensorHandle {
 public:
  TensorHandle() = default;
  TensorHandle(TensorShape shape, DataType dtype);

  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;
  TensorHandle(TensorHandle&&) noexcept = default;
  TensorHandle& operator=(TensorHandle&&) noexcept = default;

  void Reset(TensorShape shape, DataType dtype);
  void AttachCpuBuffer(void* data, std::size_t bytes, bool valid = true);
  void UseOwnedCpuBuffer(bool valid = false);
  void AllocatePinnedCpuBuffer(bool valid = false);
  void AllocateAclBuffer(bool valid = false);

  const TensorShape& shape() const { return shape_; }
  DataType dtype() const { return dtype_; }
  std::size_t bytes() const;
  TensorResidency residency() const;
  TensorHandleState State() const;
  std::string ResidencyString() const;

  bool has_cpu_buffer() const;
  bool has_pinned_cpu_buffer() const;
  bool has_acl_buffer() const;
  bool cpu_valid() const { return cpu_valid_; }
  bool pinned_cpu_valid() const { return pinned_cpu_valid_; }
  bool acl_valid() const { return acl_valid_; }
  bool cpu_dirty() const { return cpu_dirty_; }
  bool pinned_cpu_dirty() const { return pinned_cpu_dirty_; }
  bool acl_dirty() const { return acl_dirty_; }
  bool ready_event_recorded() const { return ready_event_recorded_; }
  const std::string& last_materialization_reason() const { return last_materialization_reason_; }

  TensorView CpuView() const;
  TensorView PinnedCpuView() const;
  TensorView AclView() const;

  void MarkCpuModified();
  void MarkPinnedCpuModified();
  void MarkAclModified();
  void MarkCpuValid();
  void MarkPinnedCpuValid();
  void MarkAclValid();

  TensorCopyResult EnsureAcl(void* stream = nullptr);
  TensorCopyResult MaterializeToCpu(std::string_view reason, void* stream = nullptr);

  void RecordReady(void* stream);
  void WaitReady(void* stream) const;

 private:
  void* CpuData() const;
  void* PinnedCpuData() const;
  void CheckNoConflictingDirtyState() const;

  TensorShape shape_;
  DataType dtype_ = DataType::kFloat16;
  HostBuffer cpu_buffer_;
  void* external_cpu_data_ = nullptr;
  std::size_t external_cpu_bytes_ = 0;
  bool owns_cpu_buffer_ = false;
  PinnedHostBuffer pinned_cpu_buffer_;
  AclBuffer acl_buffer_;
  bool cpu_valid_ = false;
  bool pinned_cpu_valid_ = false;
  bool acl_valid_ = false;
  bool cpu_dirty_ = false;
  bool pinned_cpu_dirty_ = false;
  bool acl_dirty_ = false;
  std::optional<int> ready_event_;
  bool ready_event_recorded_ = false;
  std::string last_materialization_reason_;
};

const char* TensorResidencyName(TensorResidency residency);

}  // namespace mlvc

#endif  // MLVC_CORE_TENSOR_HANDLE_H_
