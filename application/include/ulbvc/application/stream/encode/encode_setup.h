#ifndef ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_SETUP_H_
#define ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_SETUP_H_

#include <ulbvc/application/stream/ulbvc_internal.h>
#include <ulbvc/application/stream/stream_encoder.h>
#include <ulbvc/io/ulbvc_bitstream.h>

namespace ulbvc::codec {

void ValidateEncodeInput(const EncodeStreamOptions& options);
ulbvc::io::UlbvcBitstreamHeader BuildEncodeHeader(const EncodeStreamOptions& options,
                                                const SourceFrameGeometry& geometry, double fps);

}  // namespace ulbvc::codec

#endif  // ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_SETUP_H_
