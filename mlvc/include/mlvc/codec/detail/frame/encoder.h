#ifndef MLVC_CODEC_DETAIL_ENCODER_H_
#define MLVC_CODEC_DETAIL_ENCODER_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/entropy/entropy_codec.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/framework/async_plan.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/runtime/stage_runtime.h>

#include <cstdint>
#include <vector>

#include <mlvc/codec/detail/quantization/qscale_cache.h>
#include <mlvc/codec/detail/frame/reference_state.h>
#include <mlvc/codec/detail/memory/scratch.h>

namespace mlvc::codec {

void EncodeIAsync(mlvc::StageModelSet* models, const mlvc::RuntimeSidecar& sidecar,
                  mlvc::EntropyEncoder* entropy, mlvc::AsyncFramePlan* plan, QScaleCache* q_scales,
                  const TensorData& frame, TensorData* x_hat, mlvc::TensorHandle* x_hat_handle,
                  bool use_two_parts, AsyncEntropyScratch* entropy_scratch,
                  std::vector<uint8_t>* payload_output, mlvc::Profiler* profiler);
void EncodePAsync(mlvc::StageModelSet* models, const mlvc::RuntimeSidecar& sidecar,
                  mlvc::EntropyEncoder* entropy, mlvc::AsyncFramePlan* plan, QScaleCache* q_scales,
                  const TensorData& frame, const ReferenceState& state, int qp, bool use_two_parts,
                  AsyncEntropyScratch* entropy_scratch, TensorData* decoded_feature,
                  mlvc::TensorHandle* decoded_feature_handle, CodecScratch* scratch,
                  std::vector<uint8_t>* payload_output, mlvc::Profiler* profiler);

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_ENCODER_H_
