#ifndef MLVC_CORE_BUFFER_H_
#define MLVC_CORE_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mlvc/core/tensor.h"
#include "mlvc/core/types.h"

namespace mlvc {

class HostBuffer {
 public:
  HostBuffer() = default;
  explicit HostBuffer(std::size_t bytes);

  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
  HostBuffer(HostBuffer&&) noexcept = default;
  HostBuffer& operator=(HostBuffer&&) noexcept = default;

  void Resize(std::size_t bytes);
  void* data() { return data_.data(); }
  const void* data() const { return data_.data(); }
  std::size_t bytes() const { return data_.size(); }

  TensorView View(const TensorShape& shape, DataType dtype);

 private:
  std::vector<std::uint8_t> data_;
};

class DeviceHostBuffer {
 public:
  DeviceHostBuffer() = default;
  explicit DeviceHostBuffer(std::size_t bytes);
  ~DeviceHostBuffer();

  DeviceHostBuffer(const DeviceHostBuffer&) = delete;
  DeviceHostBuffer& operator=(const DeviceHostBuffer&) = delete;
  DeviceHostBuffer(DeviceHostBuffer&& other) noexcept;
  DeviceHostBuffer& operator=(DeviceHostBuffer&& other) noexcept;

  void Allocate(std::size_t bytes);
  void Reset();
  void* data() { return data_; }
  const void* data() const { return data_; }
  std::size_t bytes() const { return bytes_; }

  TensorView View(const TensorShape& shape, DataType dtype);

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void Allocate(std::size_t bytes);
  void Reset();
  void* data() { return data_; }
  const void* data() const { return data_; }
  std::size_t bytes() const { return bytes_; }

  TensorView View(const TensorShape& shape, DataType dtype);

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

using PinnedHostBuffer = DeviceHostBuffer;
using AclBuffer = DeviceBuffer;

}  // namespace mlvc

#endif  // MLVC_CORE_BUFFER_H_
