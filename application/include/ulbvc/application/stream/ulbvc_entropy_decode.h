#ifndef ULBVC_APPLICATION_STREAM_ULBVC_ENTROPY_DECODE_H_
#define ULBVC_APPLICATION_STREAM_ULBVC_ENTROPY_DECODE_H_

#include <ulbvc/application/stream/ulbvc_internal.h>
#include <ulbvc/entropy/ulbvc_official_entropy.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/runtime/model_manifest.h>

#include <cstdint>
#include <vector>

namespace ulbvc::codec {

DecodedEntropyFrame DecodeUlbvcEntropyFrame(ulbvc::UlbvcOfficialEntropyDecoder* decoder,
                                           const ulbvc::ModelRecord& decoder_record, int frame_index,
                                           UlbvcFrameType frame_type, int q_index,
                                           std::vector<uint8_t> payload, ulbvc::Profiler* profiler);

}  // namespace ulbvc::codec

#endif  // ULBVC_APPLICATION_STREAM_ULBVC_ENTROPY_DECODE_H_
