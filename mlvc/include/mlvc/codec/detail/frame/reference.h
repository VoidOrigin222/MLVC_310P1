#ifndef MLVC_CODEC_DETAIL_REFERENCE_H_
#define MLVC_CODEC_DETAIL_REFERENCE_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/runtime/stage_runtime.h>

#include <mlvc/codec/detail/quantization/qscale_cache.h>
#include <mlvc/codec/detail/frame/reference_state.h>

namespace mlvc::codec {

ReferenceFeatureBinding ReferenceFeature(mlvc::StageModelSet* models, const ReferenceState& state,
                                         TensorData* scratch, mlvc::Profiler* profiler);
void MaterializeReferenceFrame(mlvc::StageModelSet* models, QScaleCache* q_scales, int last_qp,
                               ReferenceState* state, TensorData* scratch,
                               mlvc::Profiler* profiler = nullptr);

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_REFERENCE_H_
