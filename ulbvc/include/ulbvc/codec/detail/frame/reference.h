#ifndef ULBVC_CODEC_DETAIL_REFERENCE_H_
#define ULBVC_CODEC_DETAIL_REFERENCE_H_

#include <ulbvc/codec/detail/frame/reference_state.h>
#include <ulbvc/codec/detail/quantization/qscale_cache.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/runtime/stage_runtime.h>

namespace ulbvc::codec {

ReferenceFeatureBinding ReferenceFeature(ulbvc::StageModelSet* models, const ReferenceState& state,
                                         TensorData* scratch, ulbvc::Profiler* profiler);
void MaterializeReferenceFrame(ulbvc::StageModelSet* models, QScaleCache* q_scales, int last_qp,
                               ReferenceState* state, TensorData* scratch,
                               ulbvc::Profiler* profiler = nullptr);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_REFERENCE_H_
