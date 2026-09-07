#ifndef MLVC_CODEC_DETAIL_DECODER_H_
#define MLVC_CODEC_DETAIL_DECODER_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/entropy/entropy_codec.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/runtime/stage_runtime.h>

#include <cstdint>
#include <vector>

#include <mlvc/codec/detail/quantization/qscale_cache.h>
#include <mlvc/codec/detail/frame/reference_state.h>
#include <mlvc/codec/detail/memory/scratch.h>

namespace mlvc::codec {

void DecodeI(mlvc::StageModelSet* models, const mlvc::RuntimeSidecar& sidecar,
             mlvc::EntropyDecoder* entropy, QScaleCache* q_scales,
             const std::vector<uint8_t>& bitstream, int qp, const SourceFrameGeometry& geometry,
             CodecScratch* scratch, TensorData* x_hat, mlvc::Profiler* profiler = nullptr);
void DecodeP(mlvc::StageModelSet* models, const mlvc::RuntimeSidecar& sidecar,
             mlvc::EntropyDecoder* entropy, QScaleCache* q_scales,
             const std::vector<uint8_t>& bitstream, const ReferenceState& state, int qp,
             const SourceFrameGeometry& geometry, CodecScratch* scratch, TensorData* x_hat,
             TensorData* decoded_feature, mlvc::TensorHandle* decoded_feature_handle,
             mlvc::Profiler* profiler = nullptr);

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_DECODER_H_
