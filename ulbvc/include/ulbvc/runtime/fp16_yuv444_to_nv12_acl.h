#ifndef ULBVC_RUNTIME_FP16_YUV444_TO_NV12_ACL_H_
#define ULBVC_RUNTIME_FP16_YUV444_TO_NV12_ACL_H_

#include <acl/acl.h>
#include <ulbvc/core/tensor.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/io/fp16_yuv444_to_nv12.h>

namespace ulbvc {

bool Fp16Yuv444ToNv12AclAvailable();

void Fp16Yuv444ToNv12Acl(const void* input_fp16, const TensorShape& input_shape,
                         const io::Nv12Layout& layout, void* output_nv12, void* stream,
                         Profiler* profiler = nullptr);
void Fp16Yuv444ToNv12Acl(const void* input_fp16, const TensorShape& input_shape,
                         const io::Nv12Layout& layout, void* output_nv12, void* stream,
                         aclrtEvent start_event, aclrtEvent ready_event,
                         Profiler* profiler = nullptr);

}  // namespace ulbvc

#endif  // ULBVC_RUNTIME_FP16_YUV444_TO_NV12_ACL_H_
