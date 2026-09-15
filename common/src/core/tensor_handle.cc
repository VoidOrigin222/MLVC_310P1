#include "mlvc/core/tensor_handle.h"

#include <acl/acl.h>

#include <cstring>
#include <utility>

#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc {
namespace {

std::size_t TensorByteSize(const TensorShape& shape, DataType dtype) {
  return shape.NumElements() * ElementSize(dtype);
}

void CheckAclStatus(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

}  // namespace

TensorHandle::TensorHandle(TensorShape shape, DataType dtype) { Reset(std::move(shape), dtype); }

void TensorHandle::Reset(TensorShape shape, DataType dtype) {
  shape_ = std::move(shape);
  dtype_ = dtype;
  cpu_buffer_ = HostBuffer();
  external_cpu_data_ = nullptr;
  external_cpu_bytes_ = 0;
  owns_cpu_buffer_ = false;
  pinned_cpu_buffer_.Reset();
  acl_buffer_.Reset();
  cpu_valid_ = false;
  pinned_cpu_valid_ = false;
  acl_valid_ = false;
  cpu_dirty_ = false;
  pinned_cpu_dirty_ = false;
  acl_dirty_ = false;
  ready_event_.reset();
  ready_event_recorded_ = false;
  last_materialization_reason_.clear();
}

void TensorHandle::AttachCpuBuffer(void* data, std::size_t bytes, bool valid) {
  Check(data != nullptr || bytes == 0, "CPU tensor handle buffer must not be null");
  Check(bytes >= this->bytes(), "CPU tensor handle buffer is too small");
  external_cpu_data_ = data;
  external_cpu_bytes_ = bytes;
  owns_cpu_buffer_ = false;
  cpu_valid_ = valid;
  cpu_dirty_ = valid && !acl_valid_ && !pinned_cpu_valid_;
}

void TensorHandle::UseOwnedCpuBuffer(bool valid) {
  cpu_buffer_.Resize(bytes());
  external_cpu_data_ = nullptr;
  external_cpu_bytes_ = 0;
  owns_cpu_buffer_ = true;
  cpu_valid_ = valid;
  cpu_dirty_ = valid && !acl_valid_ && !pinned_cpu_valid_;
}

void TensorHandle::AllocatePinnedCpuBuffer(bool valid) {
  pinned_cpu_buffer_.Allocate(bytes());
  pinned_cpu_valid_ = valid;
  pinned_cpu_dirty_ = valid && !acl_valid_ && !cpu_valid_;
}

void TensorHandle::AllocateAclBuffer(bool valid) {
  acl_buffer_.Allocate(bytes());
  acl_valid_ = valid;
  acl_dirty_ = valid && !cpu_valid_ && !pinned_cpu_valid_;
}

std::size_t TensorHandle::bytes() const { return TensorByteSize(shape_, dtype_); }

TensorResidency TensorHandle::residency() const {
  const bool has_cpu = has_cpu_buffer();
  const bool has_pinned = has_pinned_cpu_buffer();
  const bool has_acl = has_acl_buffer();
  if (!has_cpu && !has_pinned && !has_acl) {
    return TensorResidency::kEmpty;
  }
  if (has_pinned && has_acl) {
    return TensorResidency::kPinnedCpuAndAcl;
  }
  if (has_cpu && has_acl) {
    return TensorResidency::kCpuAndAcl;
  }
  if (has_acl) {
    return TensorResidency::kAcl;
  }
  if (has_pinned) {
    return TensorResidency::kPinnedCpu;
  }
  return TensorResidency::kCpu;
}

TensorHandleState TensorHandle::State() const {
  return TensorHandleState{
      shape_.dims(),
      dtype_,
      residency(),
      bytes(),
      has_cpu_buffer(),
      has_pinned_cpu_buffer(),
      has_acl_buffer(),
      cpu_valid_,
      pinned_cpu_valid_,
      acl_valid_,
      cpu_dirty_,
      pinned_cpu_dirty_,
      acl_dirty_,
      ready_event_recorded_,
      last_materialization_reason_,
  };
}

std::string TensorHandle::ResidencyString() const { return TensorResidencyName(residency()); }

bool TensorHandle::has_cpu_buffer() const {
  return (external_cpu_data_ != nullptr && external_cpu_bytes_ >= bytes()) ||
         (owns_cpu_buffer_ && cpu_buffer_.bytes() >= bytes());
}

bool TensorHandle::has_pinned_cpu_buffer() const {
  return pinned_cpu_buffer_.data() != nullptr && pinned_cpu_buffer_.bytes() >= bytes();
}

bool TensorHandle::has_acl_buffer() const {
  return acl_buffer_.data() != nullptr && acl_buffer_.bytes() >= bytes();
}

TensorView TensorHandle::CpuView() const {
  Check(has_cpu_buffer(), "CPU tensor handle buffer is not allocated");
  return TensorView::Borrowed(CpuData(), &shape_, dtype_, MemoryLocation::kCpu);
}

TensorView TensorHandle::PinnedCpuView() const {
  Check(has_pinned_cpu_buffer(), "pinned CPU tensor handle buffer is not allocated");
  return TensorView::Borrowed(PinnedCpuData(), &shape_, dtype_, MemoryLocation::kPinnedCpu);
}

TensorView TensorHandle::AclView() const {
  Check(has_acl_buffer(), "ACL tensor handle buffer is not allocated");
  return TensorView::Borrowed(const_cast<void*>(acl_buffer_.data()), &shape_, dtype_,
                              MemoryLocation::kAcl);
}

void TensorHandle::MarkCpuModified() {
  Check(has_cpu_buffer(), "CPU tensor handle buffer is not allocated");
  cpu_valid_ = true;
  pinned_cpu_valid_ = false;
  acl_valid_ = false;
  cpu_dirty_ = true;
  pinned_cpu_dirty_ = false;
  acl_dirty_ = false;
}

void TensorHandle::MarkPinnedCpuModified() {
  Check(has_pinned_cpu_buffer(), "pinned CPU tensor handle buffer is not allocated");
  pinned_cpu_valid_ = true;
  cpu_valid_ = false;
  acl_valid_ = false;
  pinned_cpu_dirty_ = true;
  cpu_dirty_ = false;
  acl_dirty_ = false;
}

void TensorHandle::MarkAclModified() {
  Check(has_acl_buffer(), "ACL tensor handle buffer is not allocated");
  acl_valid_ = true;
  cpu_valid_ = false;
  pinned_cpu_valid_ = false;
  acl_dirty_ = true;
  cpu_dirty_ = false;
  pinned_cpu_dirty_ = false;
}

void TensorHandle::MarkCpuValid() {
  Check(has_cpu_buffer(), "CPU tensor handle buffer is not allocated");
  cpu_valid_ = true;
  cpu_dirty_ = false;
  pinned_cpu_dirty_ = false;
  acl_dirty_ = false;
}

void TensorHandle::MarkPinnedCpuValid() {
  Check(has_pinned_cpu_buffer(), "pinned CPU tensor handle buffer is not allocated");
  pinned_cpu_valid_ = true;
  pinned_cpu_dirty_ = false;
  cpu_dirty_ = false;
  acl_dirty_ = false;
}

void TensorHandle::MarkAclValid() {
  Check(has_acl_buffer(), "ACL tensor handle buffer is not allocated");
  acl_valid_ = true;
  cpu_dirty_ = false;
  pinned_cpu_dirty_ = false;
  acl_dirty_ = false;
}

TensorCopyResult TensorHandle::EnsureAcl(void*) {
  MLVC_PROFILE_RANGE_FUNCTION();
  CheckNoConflictingDirtyState();
  if (!has_acl_buffer()) {
    AllocateAclBuffer(false);
  }
  if (acl_valid_ && !acl_dirty_) {
    return TensorCopyResult{false, 0, MemoryLocation::kAcl, MemoryLocation::kAcl};
  }
  if (pinned_cpu_valid_) {
    CheckAclStatus(aclrtMemcpy(acl_buffer_.data(), bytes(), PinnedCpuData(), bytes(),
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy TensorHandle pinned CPU to ACL");
    acl_valid_ = true;
    pinned_cpu_dirty_ = false;
    acl_dirty_ = false;
    return TensorCopyResult{true, bytes(), MemoryLocation::kPinnedCpu, MemoryLocation::kAcl};
  }
  Check(cpu_valid_, "TensorHandle has no valid CPU or pinned CPU source for ACL upload");
  CheckAclStatus(
      aclrtMemcpy(acl_buffer_.data(), bytes(), CpuData(), bytes(), ACL_MEMCPY_HOST_TO_DEVICE),
      "aclrtMemcpy TensorHandle CPU to ACL");
  acl_valid_ = true;
  cpu_dirty_ = false;
  acl_dirty_ = false;
  return TensorCopyResult{true, bytes(), MemoryLocation::kCpu, MemoryLocation::kAcl};
}

TensorCopyResult TensorHandle::MaterializeToCpu(std::string_view reason, void*) {
  MLVC_PROFILE_RANGE_FUNCTION();
  CheckNoConflictingDirtyState();
  last_materialization_reason_ = std::string(reason);
  if (!has_cpu_buffer()) {
    UseOwnedCpuBuffer(false);
  }
  if (cpu_valid_ && !cpu_dirty_) {
    return TensorCopyResult{false, 0, MemoryLocation::kCpu, MemoryLocation::kCpu};
  }
  if (pinned_cpu_valid_) {
    std::memcpy(CpuData(), PinnedCpuData(), bytes());
    cpu_valid_ = true;
    cpu_dirty_ = false;
    pinned_cpu_dirty_ = false;
    acl_dirty_ = false;
    return TensorCopyResult{true, bytes(), MemoryLocation::kPinnedCpu, MemoryLocation::kCpu};
  }
  Check(acl_valid_, "TensorHandle has no valid ACL source for CPU materialization");
  CheckAclStatus(
      aclrtMemcpy(CpuData(), bytes(), acl_buffer_.data(), bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
      "aclrtMemcpy TensorHandle ACL to CPU");
  cpu_valid_ = true;
  cpu_dirty_ = false;
  acl_dirty_ = false;
  return TensorCopyResult{true, bytes(), MemoryLocation::kAcl, MemoryLocation::kCpu};
}

void TensorHandle::RecordReady(void*) { ready_event_recorded_ = true; }

void TensorHandle::WaitReady(void*) const {}

void* TensorHandle::CpuData() const {
  if (external_cpu_data_ != nullptr) {
    return external_cpu_data_;
  }
  return const_cast<void*>(cpu_buffer_.data());
}

void* TensorHandle::PinnedCpuData() const { return const_cast<void*>(pinned_cpu_buffer_.data()); }

void TensorHandle::CheckNoConflictingDirtyState() const {
  const int dirty_count = (cpu_dirty_ ? 1 : 0) + (pinned_cpu_dirty_ ? 1 : 0) + (acl_dirty_ ? 1 : 0);
  Check(dirty_count <= 1, "TensorHandle has conflicting dirty buffers");
}

const char* TensorResidencyName(TensorResidency residency) {
  switch (residency) {
    case TensorResidency::kEmpty:
      return "empty";
    case TensorResidency::kCpu:
      return "cpu";
    case TensorResidency::kPinnedCpu:
      return "pinned_cpu";
    case TensorResidency::kAcl:
      return "acl";
    case TensorResidency::kCpuAndAcl:
      return "cpu_and_acl";
    case TensorResidency::kPinnedCpuAndAcl:
      return "pinned_cpu_and_acl";
  }
  return "empty";
}

}  // namespace mlvc
