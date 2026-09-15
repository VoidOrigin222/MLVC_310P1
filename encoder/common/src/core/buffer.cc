#include "mlvc/core/buffer.h"

#include <acl/acl.h>

#include <cstring>
#include <utility>

#include "mlvc/core/status.h"

namespace mlvc {
namespace {

void CheckAclStatus(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

}  // namespace

HostBuffer::HostBuffer(std::size_t bytes) : data_(bytes) {}

void HostBuffer::Resize(std::size_t bytes) { data_.resize(bytes); }

TensorView HostBuffer::View(const TensorShape& shape, DataType dtype) {
  Check(shape.NumElements() * ElementSize(dtype) <= data_.size(),
        "host buffer view exceeds buffer");
  return TensorView::Borrowed(data(), &shape, dtype, MemoryLocation::kCpu);
}

DeviceHostBuffer::DeviceHostBuffer(std::size_t bytes) { Allocate(bytes); }

DeviceHostBuffer::~DeviceHostBuffer() { Reset(); }

DeviceHostBuffer::DeviceHostBuffer(DeviceHostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

DeviceHostBuffer& DeviceHostBuffer::operator=(DeviceHostBuffer&& other) noexcept {
  if (this != &other) {
    Reset();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

void DeviceHostBuffer::Allocate(std::size_t bytes) {
  Reset();
  if (bytes == 0) {
    return;
  }
  CheckAclStatus(aclrtMallocHost(&data_, bytes), "aclrtMallocHost");
  bytes_ = bytes;
}

void DeviceHostBuffer::Reset() {
  if (data_ != nullptr) {
    (void)aclrtFreeHost(data_);
    data_ = nullptr;
    bytes_ = 0;
  }
}

TensorView DeviceHostBuffer::View(const TensorShape& shape, DataType dtype) {
  Check(shape.NumElements() * ElementSize(dtype) <= bytes_,
        "host staging buffer view exceeds buffer");
  return TensorView::Borrowed(data(), &shape, dtype, MemoryLocation::kPinnedCpu);
}

DeviceBuffer::DeviceBuffer(std::size_t bytes) { Allocate(bytes); }

DeviceBuffer::~DeviceBuffer() { Reset(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    Reset();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

void DeviceBuffer::Allocate(std::size_t bytes) {
  Reset();
  if (bytes == 0) {
    return;
  }
  CheckAclStatus(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
  bytes_ = bytes;
}

void DeviceBuffer::Reset() {
  if (data_ != nullptr) {
    (void)aclrtFree(data_);
    data_ = nullptr;
    bytes_ = 0;
  }
}

TensorView DeviceBuffer::View(const TensorShape& shape, DataType dtype) {
  Check(shape.NumElements() * ElementSize(dtype) <= bytes_, "device buffer view exceeds buffer");
  return TensorView::Borrowed(data(), &shape, dtype, MemoryLocation::kAcl);
}

}  // namespace mlvc
