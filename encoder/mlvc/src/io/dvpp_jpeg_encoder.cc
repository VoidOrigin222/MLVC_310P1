#include <mlvc/io/dvpp_jpeg_encoder.h>

#include <acl/ops/acl_dvpp.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "mlvc/core/status.h"
#include "mlvc/runtime/acl_runtime.h"

namespace mlvc::io {

class DvppJpegEncoder::Impl {
 public:
  Impl(aclrtContext context, Nv12Layout layout, uint32_t quality)
      : context_(context), layout_(layout), input_bytes_(Nv12BufferSize(layout)) {
    Check(context_ != nullptr, "DVPP JPEG encoder requires an ACL context");
    Check(layout_.width > 0 && layout_.height > 0 && layout_.width % 2 == 0 &&
              layout_.height % 2 == 0,
          "DVPP JPEG dimensions must be positive and even");
    Check(quality >= 1 && quality <= 100, "DVPP JPEG quality must be in [1, 100]");
    try {
      CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext for DVPP JPEG");
      CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream for DVPP JPEG");
      channel_desc_ = acldvppCreateChannelDesc();
      Check(channel_desc_ != nullptr, "acldvppCreateChannelDesc failed");
      CheckAcl(acldvppCreateChannel(channel_desc_), "acldvppCreateChannel");
      channel_created_ = true;
      picture_desc_ = acldvppCreatePicDesc();
      Check(picture_desc_ != nullptr, "acldvppCreatePicDesc failed");
      jpeg_config_ = acldvppCreateJpegeConfig();
      Check(jpeg_config_ != nullptr, "acldvppCreateJpegeConfig failed");
      CheckAcl(acldvppSetJpegeConfigLevel(jpeg_config_, quality),
               "acldvppSetJpegeConfigLevel");
      CheckAcl(acldvppMalloc(&input_device_, input_bytes_), "acldvppMalloc JPEG input");
      CheckAcl(acldvppSetPicDescData(picture_desc_, input_device_),
               "acldvppSetPicDescData");
      CheckAcl(acldvppSetPicDescSize(picture_desc_, static_cast<uint32_t>(input_bytes_)),
               "acldvppSetPicDescSize");
      CheckAcl(acldvppSetPicDescFormat(picture_desc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420),
               "acldvppSetPicDescFormat NV12");
      CheckAcl(acldvppSetPicDescWidth(picture_desc_, static_cast<uint32_t>(layout_.width)),
               "acldvppSetPicDescWidth");
      CheckAcl(acldvppSetPicDescHeight(picture_desc_, static_cast<uint32_t>(layout_.height)),
               "acldvppSetPicDescHeight");
      CheckAcl(acldvppSetPicDescWidthStride(
                   picture_desc_, static_cast<uint32_t>(layout_.width_stride)),
               "acldvppSetPicDescWidthStride");
      CheckAcl(acldvppSetPicDescHeightStride(
                   picture_desc_, static_cast<uint32_t>(layout_.height_stride)),
               "acldvppSetPicDescHeightStride");
      CheckAcl(acldvppJpegPredictEncSize(picture_desc_, jpeg_config_, &output_capacity_),
               "acldvppJpegPredictEncSize");
      Check(output_capacity_ > 0, "DVPP predicted a zero-byte JPEG buffer");
      CheckAcl(acldvppMalloc(&output_device_, output_capacity_),
               "acldvppMalloc JPEG output");
    } catch (...) {
      Cleanup();
      throw;
    }
  }

  ~Impl() { Cleanup(); }

  std::vector<uint8_t> Encode(const std::vector<uint8_t>& nv12, DvppJpegTiming* timing) {
    Check(nv12.size() == input_bytes_, "DVPP JPEG input NV12 size mismatch");
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext for DVPP JPEG encode");

    const auto h2d_begin = std::chrono::steady_clock::now();
    CheckAcl(aclrtMemcpy(input_device_, input_bytes_, nv12.data(), nv12.size(),
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy DVPP JPEG H2D");
    const auto h2d_end = std::chrono::steady_clock::now();

    BindInput(input_device_);
    std::vector<uint8_t> output = EncodeBound(timing);
    if (timing != nullptr) {
      timing->h2d_ms =
          std::chrono::duration<double, std::milli>(h2d_end - h2d_begin).count();
    }
    return output;
  }

  std::vector<uint8_t> EncodeDevice(const void* nv12_device, std::size_t bytes,
                                    aclrtEvent ready_event, DvppJpegTiming* timing) {
    Check(nv12_device != nullptr, "DVPP JPEG device input must not be null");
    Check(bytes == input_bytes_, "DVPP JPEG device input NV12 size mismatch");
    Check(ready_event != nullptr, "DVPP JPEG device input requires a ready event");
    CheckAcl(aclrtSetCurrentContext(context_),
             "aclrtSetCurrentContext for DVPP JPEG device encode");
    BindInput(const_cast<void*>(nv12_device));
    CheckAcl(aclrtStreamWaitEvent(stream_, ready_event),
             "aclrtStreamWaitEvent DVPP JPEG input");
    std::vector<uint8_t> output = EncodeBound(timing);
    if (timing != nullptr) timing->h2d_ms = 0.0;
    return output;
  }

 private:
  void BindInput(void* input) {
    CheckAcl(acldvppSetPicDescData(picture_desc_, input),
             "acldvppSetPicDescData JPEG input");
    CheckAcl(acldvppSetPicDescSize(picture_desc_, static_cast<uint32_t>(input_bytes_)),
             "acldvppSetPicDescSize JPEG input");
  }

  std::vector<uint8_t> EncodeBound(DvppJpegTiming* timing) {

    uint32_t output_size = output_capacity_;
    const auto encode_begin = std::chrono::steady_clock::now();
    CheckAcl(acldvppJpegEncodeAsync(channel_desc_, picture_desc_, output_device_,
                                    &output_size, jpeg_config_, stream_),
             "acldvppJpegEncodeAsync");
    CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream DVPP JPEG");
    const auto encode_end = std::chrono::steady_clock::now();
    Check(output_size > 0 && output_size <= output_capacity_,
          "DVPP returned an invalid JPEG byte count");

    std::vector<uint8_t> output(output_size);
    const auto d2h_begin = std::chrono::steady_clock::now();
    CheckAcl(aclrtMemcpy(output.data(), output.size(), output_device_, output_size,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy DVPP JPEG D2H");
    const auto d2h_end = std::chrono::steady_clock::now();
    if (timing != nullptr) {
      timing->encode_ms =
          std::chrono::duration<double, std::milli>(encode_end - encode_begin).count();
      timing->d2h_ms =
          std::chrono::duration<double, std::milli>(d2h_end - d2h_begin).count();
    }
    return output;
  }
  void Cleanup() noexcept {
    if (context_ != nullptr) (void)aclrtSetCurrentContext(context_);
    if (output_device_ != nullptr) {
      (void)acldvppFree(output_device_);
      output_device_ = nullptr;
    }
    if (input_device_ != nullptr) {
      (void)acldvppFree(input_device_);
      input_device_ = nullptr;
    }
    if (jpeg_config_ != nullptr) {
      (void)acldvppDestroyJpegeConfig(jpeg_config_);
      jpeg_config_ = nullptr;
    }
    if (picture_desc_ != nullptr) {
      (void)acldvppDestroyPicDesc(picture_desc_);
      picture_desc_ = nullptr;
    }
    if (channel_created_) {
      (void)acldvppDestroyChannel(channel_desc_);
      channel_created_ = false;
    }
    if (channel_desc_ != nullptr) {
      (void)acldvppDestroyChannelDesc(channel_desc_);
      channel_desc_ = nullptr;
    }
    if (stream_ != nullptr) {
      (void)aclrtDestroyStream(stream_);
      stream_ = nullptr;
    }
  }

  aclrtContext context_ = nullptr;
  Nv12Layout layout_;
  std::size_t input_bytes_ = 0;
  aclrtStream stream_ = nullptr;
  acldvppChannelDesc* channel_desc_ = nullptr;
  acldvppPicDesc* picture_desc_ = nullptr;
  acldvppJpegeConfig* jpeg_config_ = nullptr;
  void* input_device_ = nullptr;
  void* output_device_ = nullptr;
  uint32_t output_capacity_ = 0;
  bool channel_created_ = false;
};

DvppJpegEncoder::DvppJpegEncoder(aclrtContext context, Nv12Layout layout, uint32_t quality)
    : impl_(std::make_unique<Impl>(context, layout, quality)) {}

DvppJpegEncoder::~DvppJpegEncoder() = default;

std::vector<uint8_t> DvppJpegEncoder::Encode(const std::vector<uint8_t>& nv12,
                                             DvppJpegTiming* timing) {
  return impl_->Encode(nv12, timing);
}

std::vector<uint8_t> DvppJpegEncoder::EncodeDevice(const void* nv12_device,
                                                   std::size_t bytes,
                                                   aclrtEvent ready_event,
                                                   DvppJpegTiming* timing) {
  return impl_->EncodeDevice(nv12_device, bytes, ready_event, timing);
}

}  // namespace mlvc::io
