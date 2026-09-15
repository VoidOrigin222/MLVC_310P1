#ifndef MLVC_IO_DVPP_JPEG_ENCODER_H_
#define MLVC_IO_DVPP_JPEG_ENCODER_H_

#include <acl/acl.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "mlvc/io/fp16_yuv444_to_nv12.h"

namespace mlvc::io {

struct DvppJpegTiming {
  double h2d_ms = 0.0;
  double encode_ms = 0.0;
  double d2h_ms = 0.0;
};

class DvppJpegEncoder {
 public:
  DvppJpegEncoder(aclrtContext context, Nv12Layout layout, uint32_t quality);
  ~DvppJpegEncoder();

  DvppJpegEncoder(const DvppJpegEncoder&) = delete;
  DvppJpegEncoder& operator=(const DvppJpegEncoder&) = delete;

  std::vector<uint8_t> Encode(const std::vector<uint8_t>& nv12,
                              DvppJpegTiming* timing = nullptr);
  std::vector<uint8_t> EncodeDevice(const void* nv12_device, std::size_t bytes,
                                    aclrtEvent ready_event,
                                    DvppJpegTiming* timing = nullptr);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mlvc::io

#endif  // MLVC_IO_DVPP_JPEG_ENCODER_H_
