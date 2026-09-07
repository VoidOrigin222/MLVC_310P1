#ifndef MLVC_APPLICATION_STREAM_ENCODE_ENCODE_SETUP_H_
#define MLVC_APPLICATION_STREAM_ENCODE_ENCODE_SETUP_H_

#include <mlvc/application/stream/stream_encoder.h>
#include <mlvc/io/mlvc_bitstream.h>
#include <mlvc/application/stream/mlvc_internal.h>

namespace mlvc::codec {

void ValidateEncodeInput(const EncodeStreamOptions& options);
mlvc::io::MlvcBitstreamHeader BuildEncodeHeader(const EncodeStreamOptions& options,
                                                const SourceFrameGeometry& geometry,
                                                double fps);

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_ENCODE_ENCODE_SETUP_H_
