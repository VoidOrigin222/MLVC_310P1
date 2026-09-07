#ifndef MLVC_RUNTIME_DECODE_PRIOR_ACL_H_
#define MLVC_RUNTIME_DECODE_PRIOR_ACL_H_

#include <cstddef>
#include <cstdint>

#include "mlvc/core/types.h"
#include "mlvc/framework/profiler.h"

namespace mlvc {

struct DecodePriorAclResult {
  std::size_t elements = 0;
  double kernel_ms = 0.0;
};

bool DecodePriorAclAvailable();

DecodePriorAclResult DecodePriorSinglePartAcl(const void* full, DataType dtype, int parts,
                                              int mask_index, int channels, int height, int width,
                                              void* part, void* stream = nullptr,
                                              Profiler* profiler = nullptr);

DecodePriorAclResult DecodePriorRestoreYAcl(const int8_t* y_symbols, const void* means_fp16,
                                            int parts, int mask_index, int channels, int height,
                                            int width, void* output_fp16, void* stream = nullptr,
                                            Profiler* profiler = nullptr);

DecodePriorAclResult DecodePriorApplyChannelQuantStepAcl(
    const void* quant_step_fp16, int quant_channels, const void* values_fp16, int channels,
    int height, int width, void* values_out_fp16, void* stream = nullptr,
    Profiler* profiler = nullptr);

DecodePriorAclResult DecodePriorInt8ToFp16Acl(const int8_t* symbols, const int64_t* shape,
                                              std::size_t rank, std::size_t count,
                                              void* output_fp16, void* stream = nullptr,
                                              Profiler* profiler = nullptr);

}  // namespace mlvc

#endif  // MLVC_RUNTIME_DECODE_PRIOR_ACL_H_
