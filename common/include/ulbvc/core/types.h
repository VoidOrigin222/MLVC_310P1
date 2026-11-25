#ifndef ULBVC_CORE_TYPES_H_
#define ULBVC_CORE_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace ulbvc {

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

}  // namespace ulbvc

#endif  // ULBVC_CORE_TYPES_H_
