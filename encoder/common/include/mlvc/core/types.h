#ifndef MLVC_CORE_TYPES_H_
#define MLVC_CORE_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlvc {

enum class DataType {
  kFloat16,
  kFloat32,
  kInt8,
  kInt16,
  kInt32,
  kUInt8,
};

enum class MemoryLocation {
  kCpu,
  kPinnedCpu,
  kAcl,
};

std::size_t ElementSize(DataType dtype);
std::string DataTypeName(DataType dtype);

}  // namespace mlvc

#endif  // MLVC_CORE_TYPES_H_
