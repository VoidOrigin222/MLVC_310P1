#ifndef MLVC_APPLICATION_STREAM_MLVC_ENTROPY_DECODE_H_
#define MLVC_APPLICATION_STREAM_MLVC_ENTROPY_DECODE_H_

#include <mlvc/application/stream/mlvc_internal.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/runtime/model_manifest.h>

#include <cstdint>
#include <vector>

namespace mlvc::codec {

DecodedEntropyFrame DecodeMlvcEntropyFrame(mlvc::MlvcOfficialEntropyDecoder* decoder,
                                           const mlvc::ModelRecord& decoder_record,
                                           int frame_index, MlvcFrameType frame_type, int q_index,
                                           std::vector<uint8_t> payload,
                                           mlvc::Profiler* profiler);

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_MLVC_ENTROPY_DECODE_H_
