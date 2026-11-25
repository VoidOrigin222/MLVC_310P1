#ifndef ULBVC_IO_DVPP_JPEG_DECODER_H_
#define ULBVC_IO_DVPP_JPEG_DECODER_H_

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace ulbvc::io {

struct DvppJpegDecodeConfig {
  int width = 1920;
  int height = 1080;
  std::size_t output_surface_count = 10;
  bool normalize_padding = true;
};

struct DvppJpegDecodeStats {
  uint64_t frames = 0, normalized_frames = 0, changed_segments = 0;
  uint64_t unsupported_frames = 0, invalid_frames = 0;
  double normalization_ms = 0;
};

class DvppJpegDecodedFrame {
 public:
  DvppJpegDecodedFrame() = default;
  DvppJpegDecodedFrame(std::shared_ptr<void> owner, std::size_t bytes, int width_stride,
                       int height_stride)
      : owner_(std::move(owner)),
        bytes_(bytes),
        width_stride_(width_stride),
        height_stride_(height_stride) {}
  void* device_data() const { return owner_.get(); }
  std::size_t bytes() const { return bytes_; }
  int width_stride() const { return width_stride_; }
  int height_stride() const { return height_stride_; }

 private:
  std::shared_ptr<void> owner_;
  std::size_t bytes_ = 0;
  int width_stride_ = 0;
  int height_stride_ = 0;
};

// Decodes one host MJPEG frame to packed NV12 bytes using the DVPP JPEGD path.
// The returned layout is visible-size NV12 with width-stride aligned to 16.
class DvppJpegDecoder {
 public:
  DvppJpegDecoder(aclrtContext context, DvppJpegDecodeConfig config);
  ~DvppJpegDecoder();

  DvppJpegDecoder(const DvppJpegDecoder&) = delete;
  DvppJpegDecoder& operator=(const DvppJpegDecoder&) = delete;

  std::vector<uint8_t> Decode(const std::vector<uint8_t>& jpeg);
  // Keeps the decoded NV12 allocation on the device for downstream ACL operators.
  DvppJpegDecodedFrame DecodeDevice(const std::vector<uint8_t>& jpeg);
  DvppJpegDecodedFrame DecodeDevice(const void* jpeg, std::size_t jpeg_bytes);
  // Read after the decoding worker has stopped; a decoder instance is single-threaded.
  const DvppJpegDecodeStats& stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ulbvc::io

#endif  // ULBVC_IO_DVPP_JPEG_DECODER_H_
