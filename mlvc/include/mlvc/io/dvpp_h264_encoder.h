#ifndef MLVC_IO_DVPP_H264_ENCODER_H_
#define MLVC_IO_DVPP_H264_ENCODER_H_

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mlvc/io/fp16_yuv444_to_nv12.h"

namespace mlvc::io {

struct DvppH264EncoderConfig {
  Nv12Layout layout;
  uint32_t fps = 30;
  uint32_t gop = 96;
  uint32_t bitrate = 8'000'000;
  uint32_t output_buffer_bytes = 0;
  bool zero_copy_input = false;
  // Opt-in motion proxy policy. The legacy RTSP path keeps per-frame restart.
  bool persistent_channel = false;
  bool single_reference = false;
  uint32_t channel = 0;
};

// Synchronous one-frame interface backed by the device MPI VENC API. The
// input may be produced by an ACL stream; ready_event establishes the
// dependency before the frame is submitted. zero_copy_input keeps the source
// surface alive through the synchronous VENC call and avoids a device copy.
class DvppH264Encoder {
 public:
  DvppH264Encoder(aclrtContext context, DvppH264EncoderConfig config);
  ~DvppH264Encoder();

  DvppH264Encoder(const DvppH264Encoder&) = delete;
  DvppH264Encoder& operator=(const DvppH264Encoder&) = delete;

  std::vector<uint8_t> EncodeDevice(const void* nv12_device, std::size_t bytes,
                                    aclrtEvent ready_event = nullptr,
                                    bool force_keyframe = false);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mlvc::io

#endif  // MLVC_IO_DVPP_H264_ENCODER_H_
