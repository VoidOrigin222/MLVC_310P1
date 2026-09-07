#include "mlvc/core/tensor.h"

#include <algorithm>

#include "mlvc/core/status.h"

namespace mlvc {

std::size_t ElementSize(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat16:
      return 2;
    case DataType::kFloat32:
      return 4;
    case DataType::kInt8:
      return 1;
    case DataType::kInt16:
      return 2;
    case DataType::kInt32:
      return 4;
    case DataType::kUInt8:
      return 1;
  }
  throw Error("unsupported data type");
}

std::string DataTypeName(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat16:
      return "float16";
    case DataType::kFloat32:
      return "float32";
    case DataType::kInt8:
      return "int8";
    case DataType::kInt16:
      return "int16";
    case DataType::kInt32:
      return "int32";
    case DataType::kUInt8:
      return "uint8";
  }
  throw Error("unsupported data type");
}

TensorShape::TensorShape(std::vector<int64_t> dims) : dims_(std::move(dims)) {
  for (int64_t dim : dims_) {
    Check(dim >= 0, "tensor dimensions must be non-negative");
  }
}

std::size_t TensorShape::NumElements() const {
  if (dims_.empty()) {
    return 0;
  }
  return std::accumulate(
      dims_.begin(), dims_.end(), static_cast<std::size_t>(1),
      [](std::size_t product, int64_t dim) { return product * static_cast<std::size_t>(dim); });
}

std::vector<int64_t> TensorShape::ContiguousStrides() const {
  std::vector<int64_t> strides(dims_.size(), 1);
  for (int index = static_cast<int>(dims_.size()) - 2; index >= 0; --index) {
    strides[index] = strides[index + 1] * dims_[index + 1];
  }
  return strides;
}

TensorView::TensorView(void* data, TensorShape shape, DataType dtype, MemoryLocation location)
    : data_(data),
      owned_shape_(std::move(shape)),
      shape_(&owned_shape_),
      dtype_(dtype),
      location_(location),
      strides_(this->shape().ContiguousStrides()) {}

TensorView::TensorView(const TensorView& other)
    : data_(other.data_),
      owned_shape_(other.shape()),
      shape_(&owned_shape_),
      dtype_(other.dtype_),
      location_(other.location_),
      strides_(other.strides_) {}

TensorView& TensorView::operator=(const TensorView& other) {
  if (this == &other) {
    return *this;
  }
  data_ = other.data_;
  owned_shape_ = other.shape();
  shape_ = &owned_shape_;
  dtype_ = other.dtype_;
  location_ = other.location_;
  strides_ = other.strides_;
  return *this;
}

TensorView::TensorView(TensorView&& other) noexcept
    : data_(other.data_),
      owned_shape_(other.shape()),
      shape_(&owned_shape_),
      dtype_(other.dtype_),
      location_(other.location_),
      strides_(std::move(other.strides_)) {
  other.data_ = nullptr;
  other.shape_ = &other.owned_shape_;
}

TensorView& TensorView::operator=(TensorView&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  data_ = other.data_;
  owned_shape_ = other.shape();
  shape_ = &owned_shape_;
  dtype_ = other.dtype_;
  location_ = other.location_;
  strides_ = std::move(other.strides_);
  other.data_ = nullptr;
  other.shape_ = &other.owned_shape_;
  return *this;
}

TensorView TensorView::Borrowed(void* data, const TensorShape* shape, DataType dtype,
                                MemoryLocation location) {
  Check(shape != nullptr, "borrowed tensor view shape must not be null");
  TensorView view;
  view.data_ = data;
  view.shape_ = shape;
  view.dtype_ = dtype;
  view.location_ = location;
  view.strides_ = shape->ContiguousStrides();
  return view;
}

}  // namespace mlvc
