#ifndef MLVC_CORE_TENSOR_H_
#define MLVC_CORE_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "mlvc/core/types.h"

namespace mlvc {

class TensorShape {
 public:
  TensorShape() = default;
  explicit TensorShape(std::vector<int64_t> dims);

  const std::vector<int64_t>& dims() const { return dims_; }
  int64_t dim(int index) const { return dims_.at(index); }
  int rank() const { return static_cast<int>(dims_.size()); }
  std::size_t NumElements() const;
  std::vector<int64_t> ContiguousStrides() const;

 private:
  std::vector<int64_t> dims_;
};

class TensorView {
 public:
  TensorView() = default;
  TensorView(void* data, TensorShape shape, DataType dtype, MemoryLocation location);
  TensorView(const TensorView& other);
  TensorView& operator=(const TensorView& other);
  TensorView(TensorView&& other) noexcept;
  TensorView& operator=(TensorView&& other) noexcept;
  static TensorView Borrowed(void* data, const TensorShape* shape, DataType dtype,
                             MemoryLocation location);

  void* data() const { return data_; }
  const TensorShape& shape() const { return *shape_; }
  DataType dtype() const { return dtype_; }
  MemoryLocation location() const { return location_; }
  const std::vector<int64_t>& strides() const { return strides_; }
  std::size_t bytes() const { return shape().NumElements() * ElementSize(dtype_); }
  bool empty() const { return data_ == nullptr || shape().NumElements() == 0; }

 private:
  void* data_ = nullptr;
  TensorShape owned_shape_;
  const TensorShape* shape_ = &owned_shape_;
  DataType dtype_ = DataType::kFloat16;
  MemoryLocation location_ = MemoryLocation::kCpu;
  std::vector<int64_t> strides_;
};

}  // namespace mlvc

#endif  // MLVC_CORE_TENSOR_H_
