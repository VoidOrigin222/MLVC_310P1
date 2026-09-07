#ifndef MLVC_RUNTIME_FP16_YUV444_TO_NV12_ACL_H_
#define MLVC_RUNTIME_FP16_YUV444_TO_NV12_ACL_H_

#include <mlvc/core/tensor.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/io/fp16_yuv444_to_nv12.h>

#include <acl/acl.h>

namespace mlvc {

bool Fp16Yuv444ToNv12AclAvailable();

void Fp16Yuv444ToNv12Acl(const void* input_fp16, const TensorShape& input_shape,
                         const io::Nv12Layout& layout, void* output_nv12,
                         void* stream, Profiler* profiler = nullptr);
void Fp16Yuv444ToNv12Acl(const void* input_fp16, const TensorShape& input_shape,
                         const io::Nv12Layout& layout, void* output_nv12,
                         void* stream, aclrtEvent start_event,
                         aclrtEvent ready_event, Profiler* profiler = nullptr);

}  // namespace mlvc

#endif  // MLVC_RUNTIME_FP16_YUV444_TO_NV12_ACL_H_
