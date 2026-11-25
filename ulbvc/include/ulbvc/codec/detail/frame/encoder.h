#ifndef ULBVC_CODEC_DETAIL_ENCODER_H_
#define ULBVC_CODEC_DETAIL_ENCODER_H_

#include <ulbvc/codec/detail/frame/reference_state.h>
#include <ulbvc/codec/detail/memory/scratch.h>
#include <ulbvc/codec/detail/quantization/qscale_cache.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/entropy/entropy_codec.h>
#include <ulbvc/entropy/sidecar.h>
#include <ulbvc/framework/async_plan.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/runtime/stage_runtime.h>

#include <cstdint>
#include <vector>

namespace ulbvc::codec {

void EncodeIAsync(ulbvc::StageModelSet* models, const ulbvc::RuntimeSidecar& sidecar,
                  ulbvc::EntropyEncoder* entropy, ulbvc::AsyncFramePlan* plan, QScaleCache* q_scales,
                  const TensorData& frame, TensorData* x_hat, ulbvc::TensorHandle* x_hat_handle,
                  bool use_two_parts, AsyncEntropyScratch* entropy_scratch,
                  std::vector<uint8_t>* payload_output, ulbvc::Profiler* profiler);
void EncodePAsync(ulbvc::StageModelSet* models, const ulbvc::RuntimeSidecar& sidecar,
                  ulbvc::EntropyEncoder* entropy, ulbvc::AsyncFramePlan* plan, QScaleCache* q_scales,
                  const TensorData& frame, const ReferenceState& state, int qp, bool use_two_parts,
                  AsyncEntropyScratch* entropy_scratch, TensorData* decoded_feature,
                  ulbvc::TensorHandle* decoded_feature_handle, CodecScratch* scratch,
                  std::vector<uint8_t>* payload_output, ulbvc::Profiler* profiler);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_ENCODER_H_
