#ifndef ULBVC_CODEC_DETAIL_DECODER_H_
#define ULBVC_CODEC_DETAIL_DECODER_H_

#include <ulbvc/codec/detail/frame/reference_state.h>
#include <ulbvc/codec/detail/memory/scratch.h>
#include <ulbvc/codec/detail/quantization/qscale_cache.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/entropy/entropy_codec.h>
#include <ulbvc/entropy/sidecar.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/runtime/stage_runtime.h>

#include <cstdint>
#include <vector>

namespace ulbvc::codec {

void DecodeI(ulbvc::StageModelSet* models, const ulbvc::RuntimeSidecar& sidecar,
             ulbvc::EntropyDecoder* entropy, QScaleCache* q_scales,
             const std::vector<uint8_t>& bitstream, int qp, const SourceFrameGeometry& geometry,
             CodecScratch* scratch, TensorData* x_hat, ulbvc::Profiler* profiler = nullptr);
void DecodeP(ulbvc::StageModelSet* models, const ulbvc::RuntimeSidecar& sidecar,
             ulbvc::EntropyDecoder* entropy, QScaleCache* q_scales,
             const std::vector<uint8_t>& bitstream, const ReferenceState& state, int qp,
             const SourceFrameGeometry& geometry, CodecScratch* scratch, TensorData* x_hat,
             TensorData* decoded_feature, ulbvc::TensorHandle* decoded_feature_handle,
             ulbvc::Profiler* profiler = nullptr);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_DECODER_H_
