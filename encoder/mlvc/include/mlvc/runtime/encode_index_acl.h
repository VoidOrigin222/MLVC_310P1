#ifndef MLVC_RUNTIME_ENCODE_INDEX_ACL_H_
#define MLVC_RUNTIME_ENCODE_INDEX_ACL_H_

#include <cstddef>
#include <cstdint>

#include "mlvc/core/types.h"
#include "mlvc/framework/profiler.h"

namespace mlvc {

struct EncodeIndexAclResult {
  std::size_t elements = 0;
  double kernel_ms = 0.0;
};

bool EncodeIndexAclAvailable();

EncodeIndexAclResult BuildIndexEncodeAcl(const void* symbols, DataType symbol_dtype,
                                         const void* scales, DataType scale_dtype,
                                         const int64_t* shape, std::size_t rank, std::size_t count,
                                         float force_zero_thres, int16_t* combined,
                                         uint8_t* keep_mask, void* stream = nullptr,
                                         Profiler* profiler = nullptr);

}  // namespace mlvc

#endif  // MLVC_RUNTIME_ENCODE_INDEX_ACL_H_
